import os
import datetime

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from curriculum import Curriculum
from schema import schema
from models import build_model
from tasks import get_task_sampler
from main_utils import init_device, get_run_id, load_pretrained_model
# from eval import get_run_metrics
from main_utils import gen_dataloader

import wandb

torch.backends.cudnn.benchmark = True


def validate_model(
        model,
        n_loops,
        model_n_dims,
        n_points,
        n_dims_truncated,
        val_size=1000,
        batch_size=64,
        task_name="linear_regression",
        family="gpt_2",
        device="cuda"):
    """
    Method for model validation. Use {task_name} generated data.
    """
    task_sampler = get_task_sampler(
        task_name=task_name,
        batch_size=batch_size,
        n_points=n_points,
        n_dims=model_n_dims,
        n_dims_truncated=n_dims_truncated,
        device=device,
        sparsity=False,
    )

    val_loader = gen_dataloader(task_sampler, val_size, batch_size)

    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            xs, ys = batch['x'].to(device), batch['y'].to(device)
            if family == 'gpt2':
                output = model(xs, ys)  # [B,]
            elif family == 'gpt2_loop':
                n_loops = n_loops  # curriculum.n_loops  # K
                y_pred_list = model(xs, ys, 0, n_loops)
                output = y_pred_list[-1]  # [B, n]
            else:
                raise NotImplementedError
            point_wise_loss = (output - ys).square().mean(dim=0)
            loss = point_wise_loss[-1] / model_n_dims
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss


def calculate_gradient_norm(model):
    total_norm = 0.0
    norm_dict = {}
    for n, p in model.named_parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        norm_dict[n] = param_norm
    total_norm = total_norm ** (1. / 2)
    return norm_dict, total_norm


def train_step(curriculum, model, xs, ys, optimizer, ctx, scaler, add_inputs_embeds, use_ctx, n_loop_window, family):
    if family in ['gpt2', 'gpt2_tying']:
        if ctx is not None:
            with ctx:
                y_pred = model(xs, ys, add_inputs_embeds=add_inputs_embeds)  # [B, n]
                # list of [B, n], length K + 1, get rid of the 0-th one
                loss = (ys - y_pred).square().mean()  # auto on both K and n (number of in context samples)
        else:
            y_pred = model(xs, ys, add_inputs_embeds=add_inputs_embeds)  # [B, n]
            # list of [B, n], length K + 1, get rid of the 0-th one
            loss = (ys - y_pred).square().mean()  # auto on both K and n (number of in context samples)
    elif family in ['gpt2_loop']:
        n_loops = curriculum.n_loops  # K
        if ctx is not None:
            with ctx:
                horizon_start = max(0, n_loops - n_loop_window)
                y_pred_list = model(xs, ys, horizon_start, n_loops)
                # list of [B, n], length K
                y_pred_arr = torch.cat(y_pred_list, dim=0)  # [B * K, n]
                y_star_arr = torch.cat([ys] * len(y_pred_list), dim=0)  # [B * K, n]
                loss = (y_star_arr - y_pred_arr).square().mean()  # auto on both K and n (number of in context samples)
                y_pred = y_pred_list[-1]  # [B, n]
        else:
            horizon_start = max(0, n_loops - n_loop_window)
            y_pred_list = model(xs, ys, horizon_start, n_loops)
            # list of [B, n], length K
            y_pred_arr = torch.cat(y_pred_list, dim=0)  # [B * K, n]
            y_star_arr = torch.cat([ys] * len(y_pred_list), dim=0)  # [B * K, n]
            loss = (y_star_arr - y_pred_arr).square().mean()  # auto on both K and n (number of in context samples)
            y_pred = y_pred_list[-1]  # [B, n]
    if use_ctx:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    norm_dict, total_norm = calculate_gradient_norm(model)
    optimizer.zero_grad(set_to_none=True)
    return loss.detach(), y_pred.detach(), total_norm, norm_dict


def train_loop(args, model, curriculum, device):
    """Method for training loop from configuration."""
    return train_without_config(
        model, curriculum, lr=args.training.learning_rate,
        add_inputs_embeds=args.training.add_inputs_embeds, task_name=args.training.task_name,
        batch_size=args.training.batch_size, n_loop_window=args.training.n_loop_window,
        model_n_dims=args.model.n_dims, train_steps=args.training.train_steps,
        family=args.model.family, experiment_name=args.wandb.name,
        out_dir=args.out_dir, do_wandb_log=False, log_every_steps=args.wandb.log_every_steps,
        use_ctx=args.training.use_ctx,
        project_name=args.wandb.project, project_notes=args.wandb.notes,
        seed=args.training.seed, weight_decay=args.training.weight_decay,
        sparsity=args.training.sparsity,save_every_steps=args.training.save_every_steps, device=device)


def train_without_config(model,
                         curriculum,
                         lr=0.0001,
                         add_inputs_embeds = False,
                         task_name="linear_regression",
                         batch_size=64,
                         n_loop_window=20,
                         model_n_dims=10,
                         train_steps=10000,
                         family="gpt2",
                         experiment_name="linear_regression_gpt_2",
                         out_dir="./results2/linear_regression_baseline",
                         do_wandb_log=False,
                         log_every_steps=100,
                         use_ctx=False,
                         project_name="base_project",
                         project_notes="",
                         seed=42,
                         weight_decay=0.0,
                         sparsity=False, save_every_steps=1000, device="gpu",):
    # TORCH 2.0 ZONE ###############################
    metrics = []
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    dtype = 'float16'  # 'bfloat16', 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    if use_ctx:
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype, cache_enabled=False)
    else:
        ctx = None
    ################################################

    if do_wandb_log:
        wandb.init(
            dir=out_dir,
            project=project_name,
            #config=args.__dict__,
            notes=project_notes,
            name=experiment_name,
            #mode="disabled" if args.debug_mode else "online",
            resume=True,
        )

    torch.manual_seed(seed)

    # model = torch.compile(model)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


    # Here the model load the pretrained model
    # args, model, optimizer, curriculum, state_path, starting_step = load_pretrained_model(
    #     args, model, optimizer, curriculum, device)

    pbar = tqdm(range(0, train_steps))
    for i in pbar:
        task_sampler = get_task_sampler(
            task_name=task_name,
            batch_size=batch_size,
            n_points=curriculum.n_points,
            n_dims=model_n_dims,
            n_dims_truncated=curriculum.n_dims_truncated,
            device=device,
            sparsity=sparsity,
        )

        real_task = task_sampler()
        xs, ys = real_task.xs.float(), real_task.ys.float()

        #loss, output, total_norm, grad_norm_dict = train_step(curriculum, model, xs, ys, optimizer, ctx, scaler)
        loss, output, total_norm, grad_norm_dict = train_step(curriculum, model, xs, ys, optimizer, ctx, scaler,
                                                              add_inputs_embeds, family=family, use_ctx=use_ctx, n_loop_window=n_loop_window)

        # EVALUATION ======================================
        point_wise_tags = list(range(curriculum.n_points))  # [0, 1, 2, ..., n-1]
        if i % log_every_steps == 0:
            point_wise_loss = (output - ys).square().mean(dim=0)  # [n,]
            epoch_metrics_dict = {
                "scaled_loss": loss / curriculum.n_dims_truncated,
                "overall_loss": loss,
                "loop_times": curriculum.n_loops,
                "grad_norm/layerwise": grad_norm_dict,
                "grad_norm": total_norm,
                "pointwise/loss": dict(
                    zip(point_wise_tags, point_wise_loss.detach().cpu().numpy())
                ),
                "n_points": curriculum.n_points,
                "n_dims": curriculum.n_dims_truncated,
                "lr": optimizer.param_groups[0]['lr'],
            }

            if do_wandb_log:
                wandb.log(epoch_metrics_dict, step=i)

            # Save metrics to wandb and out metrics array
            epoch_metrics_dict["step"] = i
            metrics.append(epoch_metrics_dict)

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % save_every_steps == 0:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, os.path.join(out_dir, f"model_{i}.pt"))
    return metrics

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Running with: {args}")

    device = init_device(args)

    if args.debug_mode:
        args.out_dir = "./results/debug"

    run_id = args.training.resume_id
    if run_id is None:
        run_id = get_run_id(args)

    out_dir = os.path.join(args.out_dir, run_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir
    # add a timestamp here, if resumed, this will be the resumed time
    args.wandb['timestamp'] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    model = build_model(args.model)

    curriculum = Curriculum(args.training.curriculum)

    train_loop(args, model, curriculum, device)
