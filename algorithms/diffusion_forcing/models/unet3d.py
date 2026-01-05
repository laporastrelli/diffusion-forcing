from functools import partial
from typing import Optional, Literal
import torch
from torch import nn
import math

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from .embeddings import Timesteps, TimestepEmbedding
from .attention import SpatialAttentionBlock, TemporalAttentionBlock
from .resnet import ResnetBlock, Downsample, Upsample
from .utils import default
import sys

class NoiseLevelSequential(nn.Sequential):
    """
    Sequential module that passes the noise level to each module in the sequence if it accepts it.
    """

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor):
        for module in self:
            if isinstance(module, ResnetBlock):
                x = module(x, noise_level)
            else:
                x = module(x)
        return x


class Unet3D_original(nn.Module):
    # Special thanks to lucidrains for the implementation of the base Diffusion model
    # https://github.com/lucidrains/denoising-diffusion-pytorch

    def __init__(
        self,
        dim: int,
        x_shape: tuple[int, int, int],
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        external_cond_dim: Optional[int] = None,
        channels=3,
        resnet_block_groups=8,
        dim_mults=[1, 2, 4, 8],
        attn_resolutions=[1, 2, 4, 8],
        attn_dim_head=32,
        attn_heads=4,
        use_linear_attn=True,
        use_init_temporal_attn=True,
        init_kernel_size=7,
        is_causal=True,
        time_emb_type: Literal["sinusoidal", "rotary"] = "rotary",
        upscale=True
    ):
        super().__init__()
        self.channels = channels
        if external_cond_dim:
            raise NotImplementedError("External conditioning not yet implemented")
        self.external_cond_dim = external_cond_dim
        init_dim = default(init_dim, dim)
        out_dim = default(out_dim, channels)
        self.is_causal = is_causal
        dim_mults = list(dim_mults)

        # modified to take into account the input shape
        if x_shape[0] == 3: # assuming that when x_shape[0] == 3, 
                            # it is a video with 3 channels and has
                            # shape that supports the original code
            dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # Original line
        else:
            _, H, W = x_shape
            max_spatial_levels = int(math.floor(math.log2(min(H, W))))
            kept_mults = dim_mults[:max_spatial_levels]
            dims = [dim] + [dim * m for m in kept_mults]
        
        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        self.upscale = upscale

        noise_level_emb_dim = dim * 4
        self.noise_level_pos_embedding = nn.Sequential(
            Timesteps(dim, True, 0),
            TimestepEmbedding(in_channels=dim, time_embed_dim=noise_level_emb_dim),
        )
        self.rotary_time_pos_embedding = RotaryEmbedding(dim=attn_dim_head) if time_emb_type == "rotary" else None

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            channels,
            init_dim,
            kernel_size=(1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

        self.init_temporal_attn = (
            TemporalAttentionBlock(
                dim=init_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                is_causal=is_causal,
                rotary_emb=self.rotary_time_pos_embedding,
            )
            if use_init_temporal_attn
            else nn.Identity()
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        block_klass_noise = partial(ResnetBlock, groups=resnet_block_groups, emb_dim=noise_level_emb_dim)
        spatial_attn_klass = partial(SpatialAttentionBlock, heads=attn_heads, dim_head=attn_dim_head)
        temporal_attn_klass = partial(
            TemporalAttentionBlock,
            heads=attn_heads,
            dim_head=attn_dim_head,
            is_causal=is_causal,
            rotary_emb=self.rotary_time_pos_embedding,
        )

        curr_resolution = 1

        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1
            use_attn = curr_resolution in attn_resolutions

            print(curr_resolution)
            print(attn_resolutions)
            print(f"Down block {idx}: dim_in={dim_in}, dim_out={dim_out}, is_last={is_last}, use_attn={use_attn}")

            # down_block = [
            #                [
            #                  ResnetBlock, 
            #                  ResnetBlock, 
            #                  (spatial + temporal) AttentionBlocks
            #                ], 
            #                  Downsample
            #              ]
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        NoiseLevelSequential(
                            block_klass_noise(dim_in, dim_out),
                            block_klass_noise(dim_out, dim_out),
                            (
                                spatial_attn_klass(
                                    dim_out,
                                    use_linear=use_linear_attn and not is_last,
                                )
                                if use_attn
                                else nn.Identity()
                            ),
                            temporal_attn_klass(dim_out) if use_attn else nn.Identity(),
                        ),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            curr_resolution *= 2 if not is_last else 1
        
        sys.exit()

        # Mid block = [
        #                ResnetBlock,
        #                Spatial AttentionBlock,
        #                Temporal AttentionBlock,
        #                ResnetBlock
        #             ]
        self.mid_block = NoiseLevelSequential(
            block_klass_noise(mid_dim, mid_dim),
            spatial_attn_klass(mid_dim),
            temporal_attn_klass(mid_dim),
            block_klass_noise(mid_dim, mid_dim),
        )

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == len(in_out) - 1
            use_attn = curr_resolution in attn_resolutions

            # up_block = [
            #              [
            #                ResnetBlock, 
            #                ResnetBlock, 
            #                (spatial + temporal) AttentionBlocks
            #              ], 
            #              Upsample
            #            ]
            self.up_blocks.append(
                NoiseLevelSequential(
                    block_klass_noise(dim_out * 2, dim_in),
                    block_klass_noise(dim_in, dim_in),
                    (spatial_attn_klass(dim_in, use_linear=use_linear_attn and idx > 0) if use_attn else nn.Identity()),
                    temporal_attn_klass(dim_in) if use_attn else nn.Identity(),
                    Upsample(dim_in, upscale=self.upscale) if not is_last else nn.Identity(),
                )
            )

            curr_resolution //= 2 if not is_last else 1

        self.out = nn.Sequential(block_klass(dim * 2, dim), nn.Conv3d(dim, out_dim, 1))

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        is_causal: Optional[bool] = None,
    ):
        if is_causal is not None and is_causal != self.is_causal:
            raise ValueError("is_causal must be the same as the one used during initialization")

        noise_levels = rearrange(noise_levels, "f b -> b f")
        noise_level_emb = self.noise_level_pos_embedding(noise_levels)

        x = self.init_conv(x)
        x = self.init_temporal_attn(x)
        h = x.clone()
        
        hs = []

        for block, downsample in self.down_blocks:
            h = block(h, noise_level_emb)
            hs.append(h)
            h = downsample(h)

        h = self.mid_block(h, noise_level_emb)

        for block in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, noise_level_emb)
            
        h = torch.cat([h, x], dim=1)
        return self.out(h)


class Unet3D(nn.Module):
    """
    Unified UNet3D:
      - default behavior: original pyramidal UNet with spatial down/up-sampling (time unchanged)
      - channel-only pyramid: no down/up-sampling at all; pyramid only via channel mults + skips
    """

    def __init__(
        self,
        dim: int,
        x_shape: tuple[int, int, int],
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        external_cond_dim: Optional[int] = None,
        channels=3,
        resnet_block_groups=8,
        dim_mults=[1, 2, 4, 8],
        attn_resolutions=[1, 2, 4, 8],
        attn_dim_head=32,
        attn_heads=4,
        use_linear_attn=True,
        use_init_temporal_attn=True,
        init_kernel_size=7,
        is_causal=True,
        time_emb_type: Literal["sinusoidal", "rotary"] = "rotary",
        upscale=True,
        channel_only_pyramid: Optional[bool] = False,  # <-- NEW (optional)
    ):
        # intialization of params (unchanged)
        super().__init__()
        self.channels = channels
        if external_cond_dim:
            raise NotImplementedError("External conditioning not yet implemented")
        self.external_cond_dim = external_cond_dim

        init_dim = default(init_dim, dim) # number of channels after initial conv
        out_dim = default(out_dim, channels) # number of output channels - default is set to frame channels
        self.is_causal = is_causal
        self.upscale = upscale
        dim_mults = list(dim_mults) # multiplier for number of channels at each level

        # ---------------------------
        # NEW: decide which mode
        # If not specified, auto-enable channel-only pyramid for H=W=1 (time-series style input).
        # here **self.channel_only_pyramid** is a bool indicating which mode we're in (video vs time-series).
        # ---------------------------
        _, H, W = x_shape
        if channel_only_pyramid is None:
            self.channel_only_pyramid = (H == 1 and W == 1) # auto-detect based on input shape if not specified
        else:
            self.channel_only_pyramid = False

        # ---------------------------
        # CHANGED: dims/in_out construction
        # Original code *truncates depth* when H,W are small, which kills the pyramid for H=W=1.
        # In channel-only mode we keep full depth from dim_mults (no truncation).
        # here **dims** is the list of channel dimensions at each level.
        # ---------------------------
        if self.channel_only_pyramid:
            dims = [init_dim, *map(lambda m: dim * m, dim_mults)] 
        else:
            if x_shape[0] == 3:  # original video path
                dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
            else:
                max_spatial_levels = int(math.floor(math.log2(min(H, W))))
                kept_mults = dim_mults[:max_spatial_levels]
                dims = [dim] + [dim * m for m in kept_mults]

        in_out = list(zip(dims[:-1], dims[1:])) # list of (in_dim, out_dim) pairs for each level
        mid_dim = dims[-1] # middle dimension at the bottleneck

        # ---------------------------
        # CHANGED: init_kernel_size handling
        # - original: fixed 7
        # - unified: default 7, should be set to 1 if channel-only pyramid
        # ---------------------------
        if self.channel_only_pyramid and init_kernel_size != 1:
            init_kernel_size = 1

        # ---------------------------
        # noise embedding (same)
        # here, self.noise_level_pos_embedding is the module that embeds noise levels.
        # ---------------------------
        noise_level_emb_dim = dim * 4
        self.noise_level_pos_embedding = nn.Sequential(
            Timesteps(dim, True, 0),
            TimestepEmbedding(in_channels=dim, time_embed_dim=noise_level_emb_dim),
        )
        self.rotary_time_pos_embedding = RotaryEmbedding(dim=attn_dim_head) if time_emb_type == "rotary" else None

        # init conv (same as original)
        # [B, C, T, H, W] -> [B, init_dim, T, H', W'] (where H',W' depend on init_kernel_size)
        # no temporal modelling is applied accross T here, only spatial convolution
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            channels,
            init_dim,
            kernel_size=(1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )
    
        # temporal attention applied after initial conv (same)
        self.init_temporal_attn = (
            TemporalAttentionBlock(
                dim=init_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                is_causal=is_causal,
                rotary_emb=self.rotary_time_pos_embedding,
            )
            if use_init_temporal_attn
            else nn.Identity()
        )

        # initialize down and up blocks (modified below)
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # intialize block classes
        block_klass = partial(
            ResnetBlock, 
            groups=resnet_block_groups,
            kernel_size=3 if not self.channel_only_pyramid else 1, 
            padding=1 if not self.channel_only_pyramid else 0
        )
        block_klass_noise = partial(
            ResnetBlock, 
            groups=resnet_block_groups, 
            emb_dim=noise_level_emb_dim,
            kernel_size=3 if not self.channel_only_pyramid else 1,
            padding=1 if not self.channel_only_pyramid else 0
        )
        spatial_attn_klass = partial(
            SpatialAttentionBlock, 
            heads=attn_heads, 
            dim_head=attn_dim_head
        ) 
        temporal_attn_klass = partial(
            TemporalAttentionBlock,
            heads=attn_heads,
            dim_head=attn_dim_head,
            is_causal=is_causal,
            rotary_emb=self.rotary_time_pos_embedding,
        ) 

        curr_resolution = 1

        # ---------------------------
        # CHANGED: down path
        # - original: includes Downsample unless last
        # - channel-only: Downsample becomes Identity
        # ---------------------------
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == len(in_out) - 1

            use_attn = curr_resolution in attn_resolutions
    
            # ---------------------------
            # CHANGED: decide whether to use spatial attention
            # - channel-only: no spatial attention at all (temporal only)
            # - default: both spatial and temporal attention
            # ---------------------------
            use_spatial_attn = (not self.channel_only_pyramid) and use_attn
            use_temporal_attn = use_attn  # keep temporal attention in both modes


            self.down_blocks.append(
                nn.ModuleList(
                    [
                        NoiseLevelSequential(
                            block_klass_noise(dim_in, dim_out),
                            block_klass_noise(dim_out, dim_out),
                            (
                                spatial_attn_klass(
                                    dim_out,
                                    use_linear=use_linear_attn and not is_last,
                                )
                                if use_spatial_attn
                                else nn.Identity()
                            ),
                            temporal_attn_klass(dim_out) if use_temporal_attn else nn.Identity(),
                        ),
                        (nn.Identity() if (self.channel_only_pyramid or is_last) else Downsample(dim_out)),
                    ]
                )
            )

            # keep original attention-level bookkeeping
            curr_resolution *= 2 if not is_last else 1
        
        # ---------------------------
        # CHANGED: mid block
        # - original: includes spatial attention
        # - channel-only: spatial attention becomes Identity
        # ---------------------------
        self.mid_block = NoiseLevelSequential(
            block_klass_noise(mid_dim, mid_dim),
            spatial_attn_klass(mid_dim) if (not self.channel_only_pyramid) else nn.Identity(),
            temporal_attn_klass(mid_dim),
            block_klass_noise(mid_dim, mid_dim),
        )

        # ---------------------------
        # CHANGED: up path
        # - original: includes Upsample unless last 
        #             includes spatial attention
        # - channel-only: Upsample becomes Identity
        #                 spatial attention becomes Identity
        # ---------------------------
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == len(in_out) - 1

            use_attn = curr_resolution in attn_resolutions
            # ---------------------------
            # CHANGED: decide whether to use spatial attention
            # - channel-only: no spatial attention at all (temporal only)
            # - default: both spatial and temporal attention
            # ---------------------------
            use_spatial_attn = (not self.channel_only_pyramid) and use_attn
            use_temporal_attn = use_attn  # keep temporal attention in both modes

            self.up_blocks.append(
                NoiseLevelSequential(
                    block_klass_noise(dim_out * 2, dim_in),
                    block_klass_noise(dim_in, dim_in),
                    (spatial_attn_klass(dim_in, use_linear=use_linear_attn and idx > 0) if use_spatial_attn else nn.Identity()),
                    temporal_attn_klass(dim_in) if use_temporal_attn else nn.Identity(),
                    (nn.Identity() if (self.channel_only_pyramid or is_last) else Upsample(dim_in, upscale=self.upscale)),
                )
            )

            curr_resolution //= 2 if not is_last else 1

        # ---------------------------
        # SMALL FIX (safe when init_dim==dim; correct when init_dim!=dim):
        # final concat is [h, x0] where x0 has init_dim channels
        # ---------------------------
        self.out = nn.Sequential(
            block_klass(init_dim * 2, init_dim),
            nn.Conv3d(init_dim, out_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        is_causal: Optional[bool] = None,
    ):
        if is_causal is not None and is_causal != self.is_causal:
            raise ValueError("is_causal must be the same as the one used during initialization")

        # original behavior: expects [F, B]
        noise_levels = rearrange(noise_levels, "f b -> b f")
        noise_level_emb = self.noise_level_pos_embedding(noise_levels)

        x0 = self.init_conv(x)
        x0 = self.init_temporal_attn(x0)

        h = x0
        hs = []

        for block, downsample in self.down_blocks:
            h = block(h, noise_level_emb)
            hs.append(h)
            h = downsample(h)  # Identity in channel-only mode

        h = self.mid_block(h, noise_level_emb)

        for block in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, noise_level_emb)  # includes Upsample or Identity at end

        h = torch.cat([h, x0], dim=1)
        return self.out(h)
