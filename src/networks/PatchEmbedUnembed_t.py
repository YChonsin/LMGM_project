
import torch.nn as nn


class PatchEmbed_3d(nn.Module):
    r""" Image to Patch Embedding

    Args:
        data_size (int): Image size.  Default: 224.
        block_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, data_size: tuple,
                 block_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # data_size = (data_size, data_size, data_size)
        block_size = (block_size, block_size, block_size)
        resolution = [data_size[0] // block_size[0],
                              data_size[1] // block_size[1],
                              data_size[2] // block_size[2]]
        self.data_size = data_size
        self.block_size = block_size
        self.patches_resolution = resolution
        self.num_patches = resolution[0] * resolution[1] * resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W, Z = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # B, Ph*Pw*Pz, C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W, Z = self.data_size
        if self.norm is not None:
            flops += H * W * Z * self.embed_dim
        return flops


class PatchUnEmbed_3d(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        data_size (int): Image size.  Default: 224.
        block_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, data_size: tuple,
                 block_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # data_size = (data_size, data_size, data_size)
        block_size = (block_size, block_size, block_size)
        resolution = [data_size[0] // block_size[0],
                              data_size[1] // block_size[1],
                              data_size[2] // block_size[2]]
        self.data_size = data_size
        self.block_size = block_size
        self.resolution = resolution
        self.num_patches = resolution[0] * resolution[1] * resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, L, C = x.shape  # B, Ph*Pw*Pz, C
        x = x.permute(0, 2, 1).view(B, self.embed_dim,
                                    x_size[0], x_size[1], x_size[2])
        # B, C, H, W, Z
        return x

    def flops(self):
        flops = 0
        return flops

if __name__ == '__main__':
    pass