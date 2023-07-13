import torch.nn as nn
import torch
import einops
import numpy as np
import torch.optim as optim

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=400):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

    def forward_concat(self, x):
        pos = self.pos_table[:, :x.size(1)].detach()
        x = torch.cat([ x, pos.repeat( x.shape[0], 1, 1)], dim=2)
        return x


class SleepTransformer(nn.Module):
    def __init__(self, args):
        """
        :param encoder_filters_small, encoder_filters_big: filters of CNN output, quantized by 20
        :param encs_small, encs_big:
        """
        super().__init__()
        self.args = args

        dmodel = args.dmodel  # 128
        heads = args.heads  # 8
        dim_feedforward = args.dim_feedforward  # 128
        fc_inner = args.fc_inner  # 1024
        num_classes = args.num_classes  # 5


        num_layers = 4

        enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
        self.inner_tf = nn.TransformerEncoder(enc, num_layers)

        enc = nn.TransformerEncoderLayer(dmodel, nhead=heads, dim_feedforward=dim_feedforward)
        self.outer_tf = nn.TransformerEncoder(enc, num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dmodel))

        self.inner_pos = PositionalEncoding(dmodel)
        self.outer_pos = PositionalEncoding(dmodel)

        self.fc_out = nn.Sequential(
                        nn.Linear(dmodel, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, fc_inner),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(fc_inner, num_classes)
                    )

    def forward(self, x, inits=None, return_matches=False, extract_norm=False, return_inter_reps=False, return_final_reps=False, return_order=False):
        x = x[:, :, :, 1:, :]  # mat
        x = torch.squeeze(x).float()

        x = einops.rearrange(x, "b outer f inner -> b outer inner f")
        x_shape = x.shape

        self.batch, self.outer, self.inner,  self.features = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        x = einops.rearrange(x, "b outer inner k -> (b outer) inner k")
        x = self.inner_pos(x)
        x = einops.rearrange(x, "(b outer) inner k -> b outer inner k", b = self.batch, outer = self.outer)

        cls_token_eeg = self.cls_token.repeat(x.shape[0], x.shape[1], 1, 1)
        x = torch.cat([cls_token_eeg, x], dim=2)

        x = einops.rearrange(x, "b outer inner k -> inner (b outer) k")
        x = self.inner_tf(x)
        x = einops.rearrange(x, "inner (b outer) k -> b outer inner k", outer=self.outer,  b=self.batch)

        x = x[:, :, 0].unsqueeze(dim=2)

        x = einops.rearrange(x, "b outer inner k ->(b inner) outer k")
        x = self.outer_pos(x)
        x = einops.rearrange(x, "(b inner) outer k -> b outer inner k", b = self.batch, inner = 1)

        x = einops.rearrange(x, "b outer inner k -> outer (b inner) k")
        x = self.outer_tf(x)
        x = einops.rearrange(x,"outer (b inner) k-> b outer inner k", outer =self.outer, inner =1, b=self.batch)

        x = einops.rearrange(x,"b outer inner k -> (b outer) (inner k)")
        x = self.fc_out(x)

        return x





