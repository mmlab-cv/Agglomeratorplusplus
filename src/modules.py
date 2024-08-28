# from turtle import forward
import torch
from torch import einsum, nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import math

from utils import init_weights, init_weights_relu, exists

from pl_bolts.models.self_supervised import SimCLR

from torch.utils.checkpoint import checkpoint

TOKEN_ATTEND_SELF_VALUE = -5e-4

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = 1)
        return F.gelu(gate) * x

class Conv1dWN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=True):
        super(Conv1dWN, self).__init__()
        self.conv1d_layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups, bias=bias)
        self.conv1d_layer = nn.utils.weight_norm(self.conv1d_layer)

    def forward(self, x):
        return self.conv1d_layer(x)

class ColumnNet(pl.LightningModule):
    def __init__(self, FLAGS, dim, groups, mult = 1, activation = GEGLU):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(1):
            self.layers.append(
                PreNorm(dim, Column(FLAGS, dim, groups, mult, activation))
            )

        # weights_init(self.layers)

    def forward(self, levels):
        for net in self.layers:
            levels = net(levels) #+ levels
    
        return levels

class Column(pl.LightningModule):
    def __init__(self, FLAGS, dim, groups, mult, activation= GEGLU()):
        super().__init__()
        total_dim = dim * groups
        self.net = nn.Sequential(
            Rearrange('b n l d -> b (l d) n'),
            nn.Conv1d(total_dim, total_dim * mult * 2, 1, groups = groups),
            activation(),
            nn.Conv1d(total_dim * mult, total_dim, 1, groups = groups),
            Rearrange('b (l d) n -> b n l d', l = groups)

        )
    def forward(self,x):
        return self.net(x)

class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None):
        super().__init__()
        if hdim is None:
            hdim = idim

        if odim is None:
            odim = 2 * hdim

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs),
            nn.ReLU())
    
    def forward(self, x):
        return self.layers(x)


class ConvTokenizer(pl.LightningModule):
    def __init__(self, in_channels=3, embedding_dim=64):
        super(ConvTokenizer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                      embedding_dim // 2,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=True),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.GELU(),# (inplace=True),
            nn.Conv2d(embedding_dim // 2,
                      embedding_dim // 2,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=True),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.GELU(),# (inplace=True),
            nn.Conv2d(embedding_dim // 2,
                      embedding_dim,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=True),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),# (inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3),
                         stride=(2, 2),
                         padding=(1, 1),
                         dilation=(1, 1))
        )

    def forward(self, x):
        return self.block(x)

class PreTrainedConvTokenizer(pl.LightningModule):
    def __init__(self):
        super(PreTrainedConvTokenizer, self).__init__()
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
            
        simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
                    
        self.feature_extractor = simclr

        self.feature_extractor.eval()

        self.modules = list(self.feature_extractor.encoder.children())[:-5]
    
    def forward(self, x):
        for block in self.modules:
            x = block(x)
        return x


class ConsensusAttention(pl.LightningModule):
    def __init__(self, num_patches_side, attend_self = True, local_consensus_radius = 0, dim=192, heads=1, FLAGS=None):
        super().__init__()
        self.attend_self = attend_self
        self.local_consensus_radius = local_consensus_radius
        total_dim = FLAGS.patch_dim * FLAGS.levels

        self.normalize = nn.LayerNorm(dim)

        if self.local_consensus_radius > 0:
            coors = torch.stack(torch.meshgrid(
                torch.arange(num_patches_side),
                torch.arange(num_patches_side)
            )).float()

            coors = rearrange(coors, 'c h w -> (h w) c')
            dist = torch.cdist(coors, coors)
            mask_non_local = dist > self.local_consensus_radius
            mask_non_local = rearrange(mask_non_local, 'i j -> () i j')
            self.register_buffer('non_local_mask', mask_non_local)

    def forward(self, levels):
        _, n, _, d, device = *levels.shape, levels.device
        levels_norm = self.normalize(levels)
        
        q, k, v = levels, F.normalize(levels, dim = -1), levels

        sim = einsum('b i l d, b j l d -> b l i j', q, k) * (d ** -0.5)

        if not self.attend_self:
            self_mask = torch.eye(n, device = device, dtype = torch.bool)
            self_mask = rearrange(self_mask, 'i j -> () () i j')
            sim.masked_fill_(self_mask, TOKEN_ATTEND_SELF_VALUE)

        if self.local_consensus_radius > 0:
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(self.non_local_mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b l i j, b j l d -> b i l d', attn, v)
        
        return out

class PairwiseAttention(pl.LightningModule):
    def __init__(self, FLAGS, bool_hard_attention=False):
        super().__init__()
        self.bool_hard_attention = bool_hard_attention
        self.beta = torch.nn.parameter.Parameter(torch.tensor(0.07, device=self.device), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        total_dim = FLAGS.patch_dim * FLAGS.levels

        self.normalize = nn.Sequential(
            nn.LayerNorm(FLAGS.patch_dim),
        )

    def forward(self, levels):
        _, n, _, d, device = *levels.shape, levels.device
        levels = self.normalize(levels)
        levels = rearrange(levels, 'b n l d -> b l n d')
        product = self.beta * torch.matmul(levels, levels.transpose(2, 3)) #* (d ** -0.5)

        mask = self.sigmoid(product)
        if self.bool_hard_attention:
            mask = (product > 0).float()

        expProduct = torch.exp(product) * mask
        attention = (expProduct / expProduct.sum(dim=3, keepdim=True))
        levels = (levels.unsqueeze(3) * attention.unsqueeze(4)).sum(dim=3)
        levels = rearrange(levels, 'b l n d -> b n l d')
        return levels #+ levels

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.out = nn.Sequential(
            Rearrange('b n l d -> b l d n'),
            nn.Linear(self.num_slots, 65),
            Rearrange('b l d n -> b n l d'),
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, l, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots

        inputs = rearrange(inputs, 'b n l d -> b l n d')
        
        mu = self.slots_mu.expand(b, l, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, l, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('b l i d, b l j d -> b l i j', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('b l j d, b l i j -> b l i d', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, l, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        slots = rearrange(slots, 'b l n d -> b n l d')

        return self.out(slots)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(pl.LightningModule):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(pl.LightningModule):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(pl.LightningModule):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale    # b h n d @ b h d n

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(pl.LightningModule):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None, groups = 1):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.groups = groups

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.conv1d(x, self.weight, self.bias, groups=self.groups)
        out = self.activation(out)
        return out

class SIRENlayer1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        omega_0: float,
        learn_omega_0: bool,
        bias: bool,
        groups = 1
    ):
        """
        Implements a Linear Layer of the form y = omega_0 * [W x + b] as in Sitzmann et al., 2020, Romero et al., 2021,
        where x is 1 dimensional.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
            groups=groups
        )

        self.groups=groups

        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(omega_0)
        else:
            tensor_omega_0 = torch.zeros(1)
            tensor_omega_0.fill_(omega_0)
            self.register_buffer("omega_0", tensor_omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.omega_0 * torch.nn.functional.conv1d(
            x, self.weight, self.bias, stride=1, padding=0, groups=self.groups
        )

class ContrastiveLoss(pl.LightningModule):
    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.m = m  # margin or radius

    def forward(self, y1, y2, d=0):
        # d = 0 means y1 and y2 are supposed to be same
        # d = 1 means y1 and y2 are supposed to be different

        euc_dist = torch.nn.functional.pairwise_distance(y1, y2)

        if d == 0:
            return torch.mean(torch.pow(euc_dist, 2))  # distance squared
        else:  # d == 1
            delta = self.m - euc_dist  # sort of reverse distance
            delta = torch.clamp(delta, min=0.0, max=None)
            return torch.mean(torch.pow(delta, 2))  # mean over all rows
