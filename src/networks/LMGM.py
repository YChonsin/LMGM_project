import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from src.networks.PatchEmbedUnembed_t import PatchEmbed_3d, PatchUnEmbed_3d

from torch.fft import rfftn, irfftn

from mamba_ssm import Mamba

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # print(x.shape)
        x = self.net(x)
        # print(x.shape)
        return x


class HeadBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv3d(embed_dim, embed_dim, kernel_size=kernel_size, stride=1,
                               padding=padding, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(embed_dim, embed_dim, kernel_size=kernel_size, stride=1,
                               padding=padding, bias=False)
        self.conv3 = nn.Conv3d(embed_dim, embed_dim, kernel_size=kernel_size, stride=1,
                               padding=padding, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(embed_dim, embed_dim, kernel_size=kernel_size, stride=1,
                               padding=padding, bias=False)

    def forward(self, x):
        x_id = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = x + x_id
        x_id = x
        x = self.conv3(x)
        x = self.relu2(x)
        x = self.conv4(x)
        x = x + x_id
        return x

class One3DM(nn.Module):
    def __init__(self, input_resolution, dim, mlp_dim, d_state=16, d_conv=4, expand=2, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba_block = One3DSM(
                    input_resolution=input_resolution,
                    dim=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, x_size):
        x_id = x.clone()
        x = self.norm1(x)
        x = self.mamba_block(x, x_size) + x_id

        x_id = x.clone()
        x = self.norm2(x)
        x = self.mlp(x) + x_id

        return x

class One3DSM(nn.Module):
    def __init__(self, input_resolution, dim, d_state=16, d_conv=4, expand=2, dropout = 0.):
        super().__init__()

        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )

        self.patch_embed = PatchEmbed_3d(
            data_size=input_resolution, block_size=1, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed_3d(
            data_size=input_resolution, block_size=1, in_chans=0, embed_dim=dim,
            norm_layer=None)


    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W, X = x_size

        x1 = self.mamba(x)  # B, HWX, C

        # B, C, H, W, X
        x = self.patch_unembed(x, x_size)

        x2 = x.permute(0, 1, 3, 2, 4).contiguous()  # B C W H X
        x2 = self.patch_embed(x2)  # B WHX C
        x2 = self.mamba(x2)
        x2_size = (W, H, X)
        x2 = self.patch_unembed(x2, x2_size) # B C W H X
        x2 = x2.permute(0, 1, 3, 2, 4).contiguous()  # B C H W X
        x2 = self.patch_embed(x2) # B HWX C

        x3 = x.permute(0, 1, 4, 3, 2).contiguous()  # B C X W H
        x3 = self.patch_embed(x3)  # B XWH C
        x3 = self.mamba(x3)
        x3_size = (X, W, H)
        x3 = self.patch_unembed(x3, x3_size) # B C X W H
        x3 = x3.permute(0, 1, 4, 3, 2).contiguous()  # B C H W X
        x3 = self.patch_embed(x3) # B HWX C

        res = x1 + x2 + x3

        x = self.patch_embed(x)
        x = x + res
        return x


class FourierUnit(nn.Module):
    def __init__(self, embed_dim, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv3d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        B, C, H, W, X = x.shape
        x_size = (H, W, X)

        # (B, C, H, W, X/2+1, 2)
        fft_dim = (-3, -2, -1)
        xf = rfftn(x, dim=fft_dim, norm=self.fft_norm)
        xf = torch.stack((xf.real, xf.imag), dim=-1)

        xf = xf.permute(0, 1, 5, 2, 3, 4).contiguous()  # (B, C, 2, H, W, X/2+1)
        xf_size = (xf.shape[3], xf.shape[4], xf.shape[5])
        xf = xf.view(B, C * 2, *xf_size)

        xf = self.conv_layer(xf)
        xf = self.relu(xf)

        xf = xf.view(B, C, 2, *xf_size).permute(0, 1, 3, 4, 5, 2).contiguous()  # (B, C, H, W, X/2+1, 2)

        xf = torch.complex(xf[..., 0], xf[..., 1])

        output = irfftn(xf, s=x_size, dim=fft_dim, norm=self.fft_norm)
        return output

class SpectralTransform(nn.Module):
    def __init__(self, embed_dim, last_conv=False):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.last_conv = last_conv

        self.conv1 = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.fu = FourierUnit(embed_dim // 2)

        self.conv2 = torch.nn.Conv3d(embed_dim // 2, embed_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.fu(x)
        x1 = self.conv2(x + x1)
        return x1

class DDA(nn.Module):
    '''Dual-domain-aware block'''
    def __init__(self, embed_dim, red=1):
        super(DDA, self).__init__()
        self.S = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // red, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(embed_dim // red, embed_dim, 3, 1, 1),
        )

        self.F = SpectralTransform(embed_dim)
        self.fusion = nn.Conv3d(embed_dim * 2, embed_dim, 1, 1, 0)

    def __call__(self, x, x_size):
        B, L, C = x.shape
        H, W, X = x_size
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W, X)
        x1 = self.S(x) + x
        x2 = self.F(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.fusion(x)

        x = x.view(B, C, H * W * X).permute(0, 2, 1).contiguous()
        return x


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth,
                 mlp_ratio=4.,  attn_drop=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 d_state=16, d_mamba_conv=4, mamba_expand=2,

                 use_freq=True, no_freq_type=1,):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_freq = use_freq

        # used for dense connections
        self.conv = nn.ModuleList()
        for i in range(2, 2 + depth):
            self.conv.append(nn.Conv3d(in_channels=i * dim, out_channels=dim,
                                       kernel_size=3, stride=1, padding=1))

        # build blocks

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                One3DM(
                    input_resolution=input_resolution,
                    dim=dim,
                    mlp_dim=int(mlp_ratio * dim),
                    d_state=d_state,
                    d_conv=d_mamba_conv,
                    expand=mamba_expand,
                    dropout=attn_drop,

                )
            )
            if use_freq:
                self.blocks.append(DDA(dim))
            else:
                match no_freq_type:
                    case 2:
                        self.blocks.append(
                            One3DM(
                                input_resolution=input_resolution,
                                dim=dim,
                                mlp_dim=int(mlp_ratio * dim),
                                d_state=d_state,
                                d_conv=d_mamba_conv,
                                expand=mamba_expand,
                                dropout=attn_drop,
                            )
                        )
                    case _:
                        pass

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x, x_size):
        # B, L, C

        x_list = []
        for i, blk in enumerate(self.blocks):
            if self.use_freq != 1:
                x_list.append(x)
                x = blk(x, x_size)
            else:
                if (i % 2) == 0:
                    x_list.append(x)
                x = blk(x, x_size)
                if (i % 2) == 0:
                    continue

            # B, L, C
            for tmp in x_list:
                x = torch.cat((x, tmp), dim=2)
            count = len(x_list)

            x = x.permute(0, 2, 1)
            x = x.view(x.shape[0], x.shape[1], x_size[0], x_size[1], x_size[2])
            x = self.conv[count - 1](x)
            x = x.view(x.shape[0], x.shape[1], x_size[0] * x_size[1] * x_size[2])
            x = x.permute(0, 2, 1)
            # print(x.shape)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"



class Four3DM(nn.Module):

    def __init__(self, dim, input_resolution, depth,
                 mlp_ratio=4.,  attn_drop=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4,
                 d_state=16, d_mamba_conv=4, mamba_expand=2,

                 use_freq=True, no_freq_type=1,):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         mlp_ratio=mlp_ratio,
                                         attn_drop=attn_drop,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,

                                         d_state=d_state,
                                         d_mamba_conv=d_mamba_conv,
                                         mamba_expand=mamba_expand,

                                         use_freq=use_freq,
                                         no_freq_type=no_freq_type,
                                         )

        self.conv = nn.Conv3d(dim, dim, 3, 1, 1)


        self.patch_embed = PatchEmbed_3d(
            data_size=img_size, block_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed_3d(
            data_size=img_size, block_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        # B, L, C
        x1 = x
        x = self.residual_group(x, x_size)
        x = self.patch_unembed(x, x_size)
        # B, C, H, W, Z
        x = self.conv(x)
        x = self.patch_embed(x) + x1
        return x


class LMGM_network(nn.Module):
    def __init__(self,
                 data_size: tuple,
                 patch_size=1,
                 in_chans=1,
                 embed_dim=96,
                 depths=[3, 1],
                 mlp_ratio=2.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 img_range=1.,
                 resi_connection='1conv',

                 d_state=16, d_mamba_conv=4, mamba_expand=2,

                 use_freq=False,
                 no_freq_type=1,
                 **kwargs):
        super(LMGM_network, self).__init__()
        self.data_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1, 1)

        self.data_size = data_size

        #####################################################################################################
        self.conv_first = nn.Conv3d(in_chans, embed_dim, 3, 1, 1)
        self.head = HeadBlock(embed_dim, kernel_size=5)

        #####################################################################################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # flattening
        self.patch_embed = PatchEmbed_3d(
            data_size=data_size, block_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # unflattening
        self.patch_unembed = PatchUnEmbed_3d(
            data_size=data_size, block_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            # print(num_patches)
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)


        # build four 3DM blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Four3DM(dim=embed_dim,
                            input_resolution=(patches_resolution[0],
                                           patches_resolution[1],
                                           patches_resolution[2]),
                            depth=depths[i_layer],
                            mlp_ratio=self.mlp_ratio,
                            attn_drop=attn_drop_rate,
                            norm_layer=norm_layer,
                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=data_size,
                            patch_size=patch_size,

                            d_state=d_state,
                            d_mamba_conv=d_mamba_conv,
                            mamba_expand=mamba_expand,

                            use_freq=use_freq,
                            no_freq_type=no_freq_type,
                            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.tail = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.tail = nn.Sequential(nn.Conv3d(embed_dim, embed_dim // 4, 3, 1, 1),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv3d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv3d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        self.conv_last = nn.Conv3d(embed_dim, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

        # self.freq_branch = SFB(embed_dim)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # B, C, H, W, X
        x_size = (x.shape[2], x.shape[3], x.shape[4])
        x = self.patch_embed(x)  # B, Ph*Pw*Pz, C

        if self.ape:
            # print(x.shape, self.absolute_pos_embed.shape)
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # B, L, C
        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)
        # B, C, H, W, X
        return x

    def forward(self, x):

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.data_range

        x1 = self.conv_first(x)
        # head block
        x1 = self.head(x1)
        # four 3DM blocks
        x11 = self.forward_features(x1)
        # tail block and residual connection
        x1 = self.tail(x11) + x1
        x1 = self.conv_last(x1)
        x = x + x1

        x = x / self.data_range + self.mean

        return x




def LMGM():
    model = LMGM_network(
        data_size=(32, 32, 8),
        patch_size=1,
        in_chans=1,
        embed_dim=96,
        depths=[4],
        mlp_ratio=2.,
        ape=False,
        resi_connection='3conv',

        d_state=16,
        d_mamba_conv=4,
        mamba_expand=2,

        use_freq=True,
        no_freq_type=1
    )
    return model

if __name__ == '__main__':
    # from src.utils import num_parameters
    # print(num_parameters(LMGM()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = LMGM().to(device)
    x = torch.randn(1, 1, 32, 32, 8).to(device)
    # x = torch.randn(1, 1, 32, 32, 4).to(device)
    print(x.shape)
    x = model(x)
    print(x.shape)