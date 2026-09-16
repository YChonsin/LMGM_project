import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from src.networks.PatchEmbedUnembed_t import PatchEmbed_3d, PatchUnEmbed_3d

from einops import rearrange, repeat

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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerEncoder(nn.Module):
    #
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, x_size):
        # B, L, C
        # print(x.shape)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)



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



class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads,
                 mlp_ratio=4.,  attn_drop=0.,  downsample=None, use_checkpoint=False,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # used for dense connections
        self.conv = nn.ModuleList()
        for i in range(2, 2 + depth):
            self.conv.append(nn.Conv3d(in_channels=i * dim, out_channels=dim,
                                       kernel_size=3, stride=1, padding=1))

        # build blocks

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                TransformerEncoder(dim=dim,
                   depth=1,
                   heads=num_heads,
                   dim_head=(dim // num_heads),
                   mlp_dim=int(mlp_ratio*dim),
                   dropout=attn_drop
                   )
            )


    def forward(self, x, x_size):
        # B, L, C

        x_list = []
        for i, blk in enumerate(self.blocks):

            x_list.append(x)

            x = blk(x, x_size)

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

    def __init__(self, dim, input_resolution, depth, num_heads,
                 mlp_ratio=4.,  attn_drop=0.,
                 downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4,):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         mlp_ratio=mlp_ratio,
                                         attn_drop=attn_drop,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,

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


class FAT_network(nn.Module):
    def __init__(self,
                 data_size: tuple,
                 patch_size=1,
                 in_chans=1,
                 embed_dim=96,
                 depths=[3, 1],
                 num_heads=[2, 2],
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
        super(FAT_network, self).__init__()
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
                            num_heads=num_heads[i_layer],
                            mlp_ratio=self.mlp_ratio,
                            attn_drop=attn_drop_rate,
                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=data_size,
                            patch_size=patch_size,
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




def FAT():
    model = FAT_network(
        data_size=(32, 32, 8),
        patch_size=1,
        in_chans=1,
        embed_dim=96,
        depths=[4],
        num_heads=[2],
        mlp_ratio=2.,
        ape=False,
        resi_connection='3conv',
        attn_drop_rate=0.1,
    )
    return model

if __name__ == '__main__':
    # from src.utils import num_parameters
    # print(num_parameters(LMGM()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = FAT().to(device)
    x = torch.randn(1, 1, 32, 32, 8).to(device)
    # x = torch.randn(1, 1, 32, 32, 4).to(device)
    print(x.shape)
    x = model(x)
    print(x.shape)