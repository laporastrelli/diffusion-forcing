from typing import Tuple
import torch.nn as nn
import torch
import numpy as np
from .resnet import ResBlockWrapper


class ConvRNNCell(nn.Module):
    def __init__(self, channel_multiplier, in_channels, hidden_channels, kernel_size, bias=True):
        super(ConvRNNCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.bias = bias

        if isinstance(kernel_size, Tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
            self.padding = (kernel_size // 2, kernel_size // 2)
        else:
            raise ValueError("Invalid kernel size.")

        self.x2h = nn.Conv2d(in_channels=in_channels,
                             out_channels=hidden_channels * channel_multiplier,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)

        self.h2h = nn.Conv2d(in_channels=in_channels,
                             out_channels=hidden_channels * channel_multiplier,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             bias=bias)
        self.Wc = None
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_channels)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx):
        # Inputs:
        #       input: of shape (batch_size, in_channels, height_size, width_size)
        #       hx: of shape (batch_size, hidden_channels, height_size, width_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_channels, height_size, width_size)]
        raise NotImplementedError


class Conv2dGRUCell(ConvRNNCell):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, bias=True):
        super(Conv2dGRUCell, self).__init__(3, in_channels, hidden_channels, kernel_size, bias)

    def forward(self, input, hx):
        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy


class Resnet2dGRUCell(Conv2dGRUCell):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, bias=True):
        super(Resnet2dGRUCell, self).__init__(in_channels, hidden_channels, kernel_size=kernel_size, bias=bias)

    def _build_model(self):
        self.x2h = ResBlockWrapper(
            nn.Sequential(
                nn.Conv2d(self.in_channels, 8, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, self.hidden_channels * self.channel_multiplier,
                                   3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
            ), self.in_channels, self.hidden_channels * self.channel_multiplier)

        self.h2h = ResBlockWrapper(
            nn.Sequential(
                nn.Conv2d(self.hidden_channels, 8, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, self.hidden_channels * self.channel_multiplier,
                                   3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
            ), self.hidden_channels, self.hidden_channels * self.channel_multiplier)

        self.Wc = None


######################## NEW ########################
# (NOT USED CURRENTLY)
######################## NEW ########################
class Conv2dBiGRU(nn.Module):
    """
    Bidirectional convolutional GRU over sequences [B, T, C_in, H, W].

    Internally runs two Conv2dGRUCell instances (forward and backward),
    concatenates their hidden states along the channel dim → [B, T, 2*Hc, H, W],
    then (optionally) projects back to Hc with a 1×1 Conv2d so downstream
    modules can keep assuming the original hidden size.

    Args:
        in_channels:     input channels per frame
        hidden_channels: hidden channels per direction (Hc)
        kernel_size:     int or (kh, kw)
        bias:            bool
        project_out:     if True, apply 1×1 Conv2d to map 2*Hc → hidden_channels
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 kernel_size=3,
                 bias: bool = True,
                 project_out: bool = True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.project_out = project_out

        # Reuse your existing per-step cell
        self.fwd = Conv2dGRUCell(in_channels, hidden_channels, kernel_size=kernel_size, bias=bias)
        self.bwd = Conv2dGRUCell(in_channels, hidden_channels, kernel_size=kernel_size, bias=bias)

        # Optional 1×1 projection: [2*Hc] → [Hc]
        if project_out:
            self.proj = nn.Conv2d(2 * hidden_channels, hidden_channels, kernel_size=1)
        else:
            self.proj = None

    @torch.no_grad()
    def _zeros_like_hidden(self, x):
        # x: [B, C_in, H, W] → return zeros [B, Hc, H, W]
        B, _, H, W = x.shape
        return x.new_zeros(B, self.hidden_channels, H, W)

    def forward(self, x, h0_fwd=None, h0_bwd=None, mask=None):
        """
        Args:
            x:       [B, T, C_in, H, W]
            h0_fwd:  optional [B, Hc, H, W]
            h0_bwd:  optional [B, Hc, H, W]
            mask:    optional broadcastable mask per time step:
                     [B, T, 1, H, W] or [B, T, 1, 1, 1] (1=valid, 0=ignore)

        Returns:
            y:       [B, T, H_out, H, W] where H_out = Hc (if project_out) else 2*Hc
            hT_fwd:  [B, Hc, H, W] (final forward state at t=T-1)
            hT_bwd:  [B, Hc, H, W] (final backward state at t=0)
        """
        B, T, C, H, W = x.shape
        device = x.device
        Hc = self.hidden_channels

        # Normalize mask shapes for broadcasting
        if mask is not None:
            if mask.dim() == 3:  # [B,T,1]
                mask = mask.view(B, T, 1, 1, 1)
            elif mask.dim() == 4:  # [B,T,1,1]
                mask = mask.unsqueeze(-1)
            # now either [B,T,1,1,1] or [B,T,1,H,W]

        # Forward direction
        hf = self._zeros_like_hidden(x[:, 0]) if h0_fwd is None else h0_fwd
        fwd_seq = []
        for t in range(T):
            xt = x[:, t]
            if mask is not None:
                xt = xt * mask[:, t]
            hf = self.fwd(xt, hf)
            if mask is not None:
                hf = hf * mask[:, t]
            fwd_seq.append(hf)

        # Backward direction
        hb = self._zeros_like_hidden(x[:, -1]) if h0_bwd is None else h0_bwd
        bwd_seq_rev = []
        for t in reversed(range(T)):
            xt = x[:, t]
            if mask is not None:
                xt = xt * mask[:, t]
            hb = self.bwd(xt, hb)
            if mask is not None:
                hb = hb * mask[:, t]
            bwd_seq_rev.append(hb)
        bwd_seq = list(reversed(bwd_seq_rev))

        # Time-stack and channel-concat
        fwd_seq = torch.stack(fwd_seq, dim=1)  # [B,T,Hc,H,W]
        bwd_seq = torch.stack(bwd_seq, dim=1)  # [B,T,Hc,H,W]
        y = torch.cat([fwd_seq, bwd_seq], dim=2)  # [B,T,2*Hc,H,W]

        # Optional projection back to Hc so downstream shapes stay unchanged
        if self.project_out:
            y = y.view(B * T, 2 * Hc, H, W)
            y = self.proj(y)  # [B*T,Hc,H,W]
            y = y.view(B, T, Hc, H, W)

        hT_fwd = fwd_seq[:, -1]
        hT_bwd = bwd_seq[:, 0]
        return y, hT_fwd, hT_bwd
