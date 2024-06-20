import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from timm.models.layers import DropPath


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, k_size=3, dilation=1):
        super(ConvBlock, self).__init__()
        self.init(in_channels, out_channels, stride, k_size, dilation)

    def init(self, in_channels, out_channels, stride=1, k_size=3):
        print("Function not implemented!")
        exit(-1)


class ResidualBlock(ConvBlock):
    expansion = 1  # 扩展系数

    def init(self, in_channels, out_channels, stride, k_size, dilation):
        # super(ResidualBlock, self).__init__()
        p = k_size // 2 * dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=k_size,
                               stride=stride, padding=p, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.the_bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=k_size,
                               stride=1, padding=p, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # self.shortcut = nn.Sequential()
        # 短连接方式
        # if stride != 1 or in_channels != out_channels * self.expansion:
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

    def forward(self, x):
        residual = self.shortcut(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # x = self.conv1(F.relu(self.the_bn1(x)))
        # x = self.conv2(F.relu(self.bn2(x)))
        # return x + residual

        x = x + residual
        return nn.ReLU()(x)


class RecurrentBlock(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """

    def __init__(self, out_ch, k_size, t=2, groups=1):
        super(RecurrentBlock, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, stride=1, padding=k_size // 2, bias=False, groups=groups),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return (x1)


class RRCNNBlock(ConvBlock):
    """
    Recurrent Residual Convolutional Neural Network Block
    """

    def init(self, in_channels, out_channels, stride, k_size, dilation):
        assert dilation == 1
        t = 2
        self.RCNN = nn.Sequential(
            RecurrentBlock(out_channels, k_size=k_size, t=t),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            RecurrentBlock(out_channels, k_size=k_size, t=t),
            nn.BatchNorm2d(out_channels),
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        )

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return nn.ReLU()(out)


class RecurrentConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = RecurrentBlock(dim, k_size=7, groups=dim)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = RecurrentBlock(dim, k_size=3)
        # self.pwconv1 = nn.Conv2d(dim,4* dim, kernel_size=3, padding=1)# nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = RecurrentBlock(dim, k_size=3)
        # self.pwconv2 = nn.Conv2d(4*dim, dim, kernel_size=3, padding=1)#nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@MODELS.register_module()
class FRNet(nn.Module):
    def __init__(self, ch_in, ch_out, ls_mid_ch=([32] * 6), out_k_size=11, k_size=3,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock) -> None:
        super().__init__()
        self.dict_module = nn.ModuleDict()

        ch1 = ch_in
        for i in range(len(ls_mid_ch)):
            ch2 = ls_mid_ch[i]
            # self.dict_module.add_module(f"conv{i}",nn.Sequential(
            #                         ResidualBlock(ch1,ch2, k_size=1),ResidualBlock(ch2,ch2)))
            if ch1 != ch2:
                self.dict_module.add_module(f"conv{i}", cls_init_block(ch1, ch2, k_size=k_size))
            else:
                if cls_conv_block == RecurrentConvNeXtBlock:
                    module = RecurrentConvNeXtBlock(dim=ch1, layer_scale_init_value=1, drop_path=0.4)
                else:
                    module = cls_conv_block(ch1, ch2, k_size=k_size)
                self.dict_module.add_module(f"conv{i}", module)

            # self.dict_module.add_module(f"shortcut{i}",ResidualBlock(ch1+ch2,ch2))
            ch1 = ch2

        # out_k_size = 11
        # self.dict_module.add_module(f"final", nn.Sequential(
        #     nn.Conv2d(ch1, ch_out * 4, out_k_size, padding=out_k_size // 2, bias=False),
        #     nn.Sigmoid()
        # ))

        self.ls_mid_ch = ls_mid_ch

    def forward(self, x):

        for i in range(len(self.ls_mid_ch)):
            conv = self.dict_module[f'conv{i}']
            x = conv(x)

        # x = self.dict_module['final'](x)
        # x = torch.max(x, dim=1, keepdim=True)[0]

        return x


if __name__ == '__main__':
    net = FRNet(1, 4)
    # print(net(x))
    # print(net)
    # import torch
    from torchsummary import summary

    # 使用 torchsummary 来查看模型的结构和层数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(summary(net.to(device), input_size=(2, 32, 32)))  # 输入尺寸根据你的模型输入要求来设置
