import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import numpy as np

from pl_bolts.models.self_supervised import SimCLR

from positional_encoding import SinusoidalPositionalEmbedding
from utils import ConvEmbed, default, exists, plot_BU_TD_TIME_ATT, init_weights
from modules import ColumnNet, ConsensusAttention, PreTrainedConvTokenizer, PreNorm, PairwiseAttention, SlotAttention, ConvTokenizer, SRTConvBlock

class Agglomerator(pl.LightningModule):
    def __init__(
        self,
        FLAGS,
        *,
        consensus_self = False,
        local_consensus_radius = 0
        ):
        super(Agglomerator, self).__init__()
        self.FLAGS = FLAGS

        self.patch_size = self.FLAGS.patch_size
        self.n_channels = self.FLAGS.n_channels

        self.convembdim = 8
        self.convembdimNopatch = int(self.FLAGS.image_size/(self.FLAGS.patch_size*self.FLAGS.patch_size*2))
        
        self.num_patches_side = (self.FLAGS.image_size // self.FLAGS.patch_size)
        self.num_patches = self.num_patches_side ** 2

        if(self.FLAGS.use_time_token):
            self.time_token = nn.Parameter(torch.randn(1, 1, self.FLAGS.patch_dim))
            self.FLAGS.num_tokens += 1


        if(self.FLAGS.use_token):
            self.class_token = nn.Parameter(torch.randn(1, 1, self.FLAGS.patch_dim))
            self.FLAGS.num_tokens += 1

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + self.FLAGS.num_tokens,1, self.FLAGS.patch_dim))

        self.iters = default(self.FLAGS.iters, self.FLAGS.levels*2)
        self.batch_acc = 0

        self.to_patch_embedding_conv = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.FLAGS.patch_size, p2 = self.FLAGS.patch_size),
        )

        self.image_to_tokens = nn.Sequential(
            ConvTokenizer(in_channels=self.FLAGS.n_channels, embedding_dim=self.FLAGS.patch_dim), # b d h w
            Rearrange('b d h w -> b (h w) d')
        )

        self.init_levels = nn.Parameter(torch.randn(self.FLAGS.levels, self.FLAGS.patch_dim))
        
        self.bottom_up = ColumnNet(self.FLAGS, dim = self.FLAGS.patch_dim, groups = self.FLAGS.levels )
        self.top_down = ColumnNet(self.FLAGS, dim = self.FLAGS.patch_dim, groups = self.FLAGS.levels - 1)
        self.attention = ConsensusAttention(self.num_patches_side, attend_self = consensus_self, local_consensus_radius = local_consensus_radius, dim=self.FLAGS.patch_dim, heads=1, FLAGS=FLAGS)
        
        self.iters_weights = nn.Parameter(torch.full((4, self.iters), 1.0/4.0))
        self.apply(init_weights)
        self.to_patch_embedding_conv.apply(init_weights)
        self.image_to_tokens.apply(init_weights)
        self.bottom_up.apply(init_weights)
        self.top_down.apply(init_weights)
        self.attention.apply(init_weights)

    def init_forward(self, bottom_level):
        b, device = bottom_level.shape[0], bottom_level.device
        n = bottom_level.shape[1]
        l = self.FLAGS.levels
        t = self.FLAGS.num_tokens
        
        bottom_level = rearrange(bottom_level, 'b n d -> b n () d')

        if(self.FLAGS.use_time_token):
            time_tokens = repeat(self.time_token, '() n d -> b n () d', b = b)
            bottom_level = torch.cat((time_tokens, bottom_level), dim=1)

        levels = repeat(self.init_levels, 'l d -> b n l d', b = b, n = n + t)

        self.num_contributions = torch.empty(self.FLAGS.levels, device=bottom_level.device).fill_(4)
        self.num_contributions[-1] = 3

        return bottom_level, levels

    def forward(self, bottom_level, levels, current_iter):

        levels_with_input = torch.cat((bottom_level, levels), dim = -2)

        bu_out = self.bottom_up(levels_with_input[..., :-1,:])
        td_out = F.pad(self.top_down(levels_with_input[..., 2:, :] + self.pos_embedding), (0, 0, 0, 1), value = 0.)
        consensus = self.attention(levels)
        
        levels = torch.stack((
            levels * self.iters_weights[0, current_iter],
            bu_out * self.iters_weights[1, current_iter],
            td_out * self.iters_weights[2, current_iter],
            consensus * self.iters_weights[3, current_iter],
        )).sum(dim=0)
            

        return bottom_level, levels, bu_out, td_out, levels[0]

    def positionalEncoding(self, resolution, count):
        # the basis: [pi, 2pi, 4pi, 8pi...]
        basis = (2.0 ** (torch.linspace(0, count - 1, count))) * np.pi
        basis = basis.view(1, 1, 1, -1)

        # create the grid
        edge = torch.linspace(0, 1, resolution)
        grid = torch.stack(torch.meshgrid(edge, edge), dim=2).unsqueeze(3)
        arg = (grid * basis).flatten(2, 3)
        sin, cos = torch.sin(arg), torch.cos(arg)
        return torch.cat([sin, cos], dim=2).permute(2, 0, 1)

