import torch
import torch.nn as nn
from nano_gpt import GPT2Model, GPT2Config, LayerNorm

MAX_NUM_CLASS = 2  # for openML classification task

def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            pred_type=conf.pred_type,
        )
    elif conf.family == 'gpt2_loop':
        model = TransformerModelLooped(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            loop_func=conf.loop_func,
            pred_type=conf.pred_type,
        )
    elif conf.family == 'gpt2_lastNtokens':
        model = TransformerModelLoopedLastNTokens(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            n=conf.last_n_tokens,
        )
    elif conf.family == 'gpt2_firstNtokens':
        model = TransformerModelLoopedFirstNTokens(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            n=conf.first_n_tokens,
        )
    elif conf.family == 'gpt2_tying':
        model = TransformerModelTying(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    else:
        raise NotImplementedError

    return model


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, pred_type='regression'):

        super(TransformerModel, self).__init__()
        self.freq = 2
        self.ind = 0
        configuration = GPT2Config()
        configuration.block_size = self.freq * n_positions + 1
        configuration.n_layer = n_layer
        configuration.n_head = n_head
        configuration.n_embd = n_embd
        configuration.dropout = 0.0
        configuration.bias = True
        configuration.dropout = 0.
        self.configuration = configuration

        self.n_positions = n_positions  # n = points in this setting
        self.n_dims = n_dims  # input dimension, d_in
        self.n_embd = n_embd  # d
        self.n_layer = n_layer
        self._pred_type = pred_type

        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(self.configuration)
        if self._pred_type == 'regression':
            self._read_out = nn.Linear(n_embd, 1)
        elif self._pred_type == 'classification':
            self._read_out = nn.Linear(n_embd, MAX_NUM_CLASS)  # NOTE: hard-code

        self.print_flag = False

    def _combine(self, xs_b, ys_b):
        """
        :param xs_b: shape [B, n, d_in]
        :param ys_b: shape [B, n]
        :return: shape [B, 2n, d_in + 1]
        """
        B, n, d = xs_b.shape
        device = xs_b.device

        ys_b_wide = torch.cat(
            (
                ys_b.view(B, n, 1),
                torch.zeros(B, n, d-1, device=device),
            ),
            axis=2,
        )

        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(B, self.freq * n, d)

        return zs

    def forward(self, xs, ys, add_inputs_embeds=False):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :return:
        """

        B, n, d_in = xs.shape
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]

        f_output = self._backbone(
            inputs_embeds=embeds, position_ids=None, rm_pos_embd=False, add_inputs_embeds=add_inputs_embeds)  # [B, 2n, d]
        prediction = self._read_out(f_output)  # [B, 2n, d] -> [B, 2n, 1]
        if self._pred_type == 'regression':
            y = prediction[:, self.ind::self.freq, 0]
        elif self._pred_type == 'classification':
            y = prediction[:, self.ind::self.freq]
        else:
            raise NotImplementedError

        return y


class TransformerModelTying(TransformerModel):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):

        super(TransformerModelTying, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head)

        self.configuration.n_layer = 1

        self._backbone = GPT2Model(self.configuration)

        self.print_flag = False

    def f(self, output):
        f_output = self._backbone(inputs_embeds=output)  # [B, 2n + 1, d]
        return f_output

    def forward(self, xs, ys, add_inputs_embeds):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :param n_loop_start: int
        :param n_loops: int
        :return:
        """
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]
        output = embeds  # also of shape [B, 2n, d]

        for idx in range(self.n_layer):
            output = self.f(output)
        prediction = self._read_out(output)  # [B, 2n, d] -> [B, 2n, 1]
        y = prediction[:, self.ind::self.freq, 0]  # [B, n]

        return y


class TransformerModelLooped(TransformerModel):
    def __init__(
            self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, loop_func='z=f(x+z)', pred_type='regression'):

        super(TransformerModelLooped, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head, pred_type)
        self.loop_func = loop_func

    def f(self, output, embeds):
        if self.loop_func == 'z=f(x+z)':
            f_output = self._backbone(inputs_embeds=output + embeds)  # [B, 2n + 1, d]
        elif self.loop_func == 'z=f(x*z)':
            f_output = self._backbone(inputs_embeds=output * embeds)  # [B, 2n + 1, d]
        else:
            raise NotImplementedError
        return f_output

    def forward(self, xs, ys, n_loop_start, n_loops):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :param n_loop_start: int
        :param n_loops: int
        :return:
        """
        B, n, d_in = xs.shape
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]
        if self.loop_func in ['z=f(x+z)']:
            output = torch.zeros_like(embeds)  # also of shape [B, 2n, d]
        elif self.loop_func in ['z=f(x*z)']:
            output = torch.ones_like(embeds)  # also of shape [B, 2n, d]
        else:
            raise NotImplementedError("Currently we only support loop function z=f(x+z) or z=f(x*z).")

        pred_list = []
        for idx in range(n_loops):
            if idx < n_loop_start:  # this will save memory when n_loops large.
                with torch.no_grad():
                    output = self.f(output, embeds)
            else:
                output = self.f(output, embeds)
                prediction = self._read_out(output)  # [B, 2n, d] -> [B, 2n, 1]
                if self._pred_type == 'regression':
                    y = prediction[:, self.ind::self.freq, 0]
                elif self._pred_type == 'classification':
                    y = prediction[:, self.ind::self.freq]
                else:
                    raise NotImplementedError
                pred_list.append(y)
            if not self.print_flag:
                print(idx)
                self.print_flag = True

        return pred_list


class TransformerModelLoopedLastNTokens(TransformerModelLooped):
    def __init__(self, n_dims, n_positions, n, n_embd=128,
                 n_layer=12, n_head=4, loop_func='z=f(x+z)',
                 pred_type='regression'):

        super(TransformerModelLoopedLastNTokens, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head, pred_type)
        self.loop_func = loop_func
        self.n = n

    def f(self, output, embeds):
        if not self.training:
            output = self.get_last_n_tokens(output, self.n)
        if self.loop_func == 'z=f(x+z)':
            f_output = self._backbone(inputs_embeds=output + embeds)  # [B, 2n + 1, d]
        elif self.loop_func == 'z=f(x*z)':
            f_output = self._backbone(inputs_embeds=output * embeds)  # [B, 2n + 1, d]
        else:
            raise NotImplementedError
        return f_output

    def get_last_n_tokens(self, x: torch.Tensor, n: int) -> torch.Tensor:
        # Take last n tokens from input of format [B, 2n, d]
        assert x.shape[1] - n * self.freq >= 0
        if self.loop_func == 'z=f(x+z)':
            x_mask = torch.zeros((x.shape[0], x.shape[1] - n * self.freq, x.shape[2])).cuda()
        elif self.loop_func == 'z=f(x*z)':
            x_mask = torch.ones((x.shape[0], x.shape[1] - n * self.freq, x.shape[2])).cuda()
        else:
            raise NotImplementedError

        x_n = x[:, -n * self.freq:, :]
        return torch.cat([x_mask, x_n], dim=1)


class TransformerModelLoopedFirstNTokens(TransformerModelLooped):
    def __init__(self, n_dims, n_positions, n, n_embd=128,
                 n_layer=12, n_head=4, loop_func='z=f(x+z)',
                 pred_type='regression'):

        super(TransformerModelLoopedFirstNTokens, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head, pred_type)
        self.loop_func = loop_func
        self.n = n

    def f(self, output, embeds):
        output = self.get_first_n_tokens(output, self.n)
        if self.loop_func == 'z=f(x+z)':
            f_output = self._backbone(inputs_embeds=output + embeds)  # [B, 2n + 1, d]
        elif self.loop_func == 'z=f(x*z)':
            f_output = self._backbone(inputs_embeds=output * embeds)  # [B, 2n + 1, d]
        else:
            raise NotImplementedError
        return f_output

    def get_first_n_tokens(self, x: torch.Tensor, n: int) -> torch.Tensor:
        # Take last n tokens from input of format [B, 2n, d]
        assert x.shape[1] - n * self.freq >= 0
        if self.loop_func == 'z=f(x+z)':
            x_mask = torch.zeros((x.shape[0], x.shape[1] - n * self.freq, x.shape[2])).cuda()
        elif self.loop_func == 'z=f(x*z)':
            x_mask = torch.ones((x.shape[0], x.shape[1] - n * self.freq, x.shape[2])).cuda()
        else:
            raise NotImplementedError

        x_n = x[:, :n * self.freq, :]
        return torch.cat([x_n, x_mask], dim=1)


if __name__ == '__main__':
    from train import get_task_sampler
    #transformer_model = TransformerModelLoopedLastNTokens(
    transformer_model = TransformerModelLooped(
        n_dims=3,
        n_positions=101,
        #n=3,
        n_embd=4,
        n_layer=1,
        n_head=2,
        pred_type="regression",
    ).cuda()
    task_sampler = get_task_sampler(
        task_name="linear_regression",
        batch_size=1,
        n_points=5,
        n_dims=3,
        n_dims_truncated=3,
        device="cuda"
    )

    real_task = task_sampler()
    xs, ys = real_task.xs.float(), real_task.ys.float()
    n_loops = 3  # K
    n_loop_window = 20

    horizon_start = max(0, n_loops - n_loop_window)
    ## forward pass
    out = transformer_model(xs, ys, horizon_start, n_loops)
    B, n, d_in = xs.shape
    zs = transformer_model._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]