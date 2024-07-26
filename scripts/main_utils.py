import torch
import datetime
import uuid
import os
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

class my_Dataset(Dataset):
    """This function reads the data from a pickle file and creates a PyTorch dataset, which contains:
    state, action, reward, reward-to-go, target
    """
    def __init__(self, xs, ys):
        self.xs = xs  # [N, n, d]
        self.ys = ys  # [N, n]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return {
            'x': self.xs[index].float(),  # [n, d]
            'y': self.ys[index].float()
        }


def gen_dataloader(task_sampler, num_sample, batch_size):
    from torch.utils.data import DataLoader

    xs_list, ys_list = [], []
    for i in range(num_sample // batch_size):
        task = task_sampler()
        xs, ys = task.xs.float().cpu(), task.ys.float().cpu()
        xs_list.extend(xs)
        ys_list.extend(ys)
    dataset = my_Dataset(xs_list, ys_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def init_device(args):
    cuda = args.gpu.cuda
    gpu = args.gpu.n_gpu
    if cuda:
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cpu")
        torch.set_num_threads(4)
    return device


def rm_orig_mod(state_dict):
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict


def load_pretrained_model(args, model, optimizer, curriculum, device):
    state_path = os.path.join(args.out_dir, "state.pt")
    starting_step = 0
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)  # NOTE: change to cpu if OOM
        state_dict = state["model_state_dict"]  # rm_orig_mod(state["model_state_dict"])
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()
        del state
        del state_dict
    elif args.model.pretrained_path is not None:
        state = torch.load(args.model.pretrained_path, map_location=device)
        if "model_state_dict" in state.keys():
            state_dict = state["model_state_dict"]  # rm_orig_mod(state["model_state_dict"])
            model.load_state_dict(state_dict, strict=False)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            for i in range(state["train_step"] + 1):
                curriculum.update()
            starting_step = state["train_step"]
            del state
            del state_dict
        else:
            state_dict = rm_orig_mod(state["model"])
            model.load_state_dict(state_dict)

            def find_train_step(s):
                step = s[s.find('model_') + 6:s.find('.pt')]
                return int(step)

            num_train_step = find_train_step(args.model.pretrained_path)
            starting_step = num_train_step
            for i in range(num_train_step + 1):
                curriculum.update()
            del state
            del state_dict
    else:
        print("train from scratch")
    return args, model, optimizer, curriculum, state_path, starting_step


def get_run_id(args):
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    run_id = "{}-{}-".format(now, args.wandb.name) + str(uuid.uuid4())[:4]
    return run_id

class GraphPlotter:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 4))
        self.ax = self.fig.subplots(1, 3)

    def plot_graph(self, metrics):
        steps = [t_loss["step"] for t_loss in metrics]
        self.ax[0].plot(steps, [t_loss["overall_loss"] for t_loss in metrics])
        self.ax[0].legend(['Train'])

        self.ax[1].plot(steps, [list(t_loss["pointwise/loss"].values())[-1] for t_loss in metrics])
        self.ax[1].legend(['pointwise_loss_mean'])

        self.ax[2].plot(steps, [t_loss["scaled_loss"] for t_loss in metrics])
        self.ax[2].legend(['scaled_loss'])

        plt.show()
