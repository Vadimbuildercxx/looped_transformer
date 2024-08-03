"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

Reference Nano-GPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from mamba_block import MambaBlock

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


@dataclass
class MambaConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 5 # Mamba depth
    dt_rank: int = 16
    dim_inner: int = 256
    d_state: int = 16
    d_conv: int = 4
    n_embd: int = 768
    img_dim: int = 64
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class Mamba(nn.Module):
    """Mamba model.

    Args:
        vocab_size (int): The size of the vocabulary.
        dim (int): The input dimension.
        depth (int): The depth of the Mamba block.
        d_state (int): The state dimension. Default is 16.
        expand (int): The expansion factor. Default is 2.
        dt_rank (Union[int, str]): The rank of the temporal difference (Δ) tensor. Default is "auto".
        d_conv (int): The dimension of the convolutional kernel. Default is 4.

    Examples:
    x = torch.randint(0, 16, (1, 64))
    model = Mamba(16, 64, 5, 16)
    out = model(x)
    print(out)
    """

    def __init__(
            self,
            config,
    ):
        """Full Mamba model."""
        super(Mamba, self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.mamba_layers = nn.ModuleList(
            [
                MambaBlock(
                    dim=config.n_embd, depth=config.n_layer, d_state=config.d_state, d_conv=config.d_conv
                )
                for _ in range(config.n_layer)
            ]
        )
        # Projection for img
        self.img_proj = nn.Linear(config.img_dim, config.n_embd)
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor = None,
    ):
        """
        Args:
            x (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(x)

        if exists(context):
            # Project the image
            projected_img = self.img_proj(context)

            # Concatenate the image and text
            x = torch.cat([x, projected_img], dim=1)

        for layer in self.mamba_layers:
            x = layer(self.norm_f(x)) + x

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits


def exists(val):
    """
    Check if the value is not None.

    Args:
        val: The value to check.

    Returns:
        bool: True if value exists (is not None), False otherwise.
    """
    return val is not None


class RMSNorm(nn.Module):
    """
    RMS  Normalization

    Args:
        dim (int): The dimension of the input.
        eps (float): The epsilon value.

    Attributes:
        scale (float): The scale value.
        gamma (nn.Parameter): The gamma parameter.

    Example:
        >>> module = RMSNorm(768)
        >>> x = torch.randn(2, 197, 768)
        >>> y = module(x)
        >>> y.shape
        torch.Size([2, 197, 768])

    """

    def __init__(self, dim, groups=1):
        super().__init__()
        self.scale = dim**-0.5
        self.gamma = nn.Parameter(torch.ones(groups, dim, 1))

    def forward(self, x):
        """Forward method implementation."""
        normed = F.normalize(x, dim=-2)
        return normed * self.scale * self.gamma