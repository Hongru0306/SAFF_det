import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter




def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class BConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)

        return x6

class Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)

        return x6



class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class DSM_SpatialGate(nn.Module):
    def __init__(self, channel):
        super(DSM_SpatialGate, self).__init__()
        kernel_size = 1
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, 3, padding=1)
        self.dw1 = nn.Sequential(
            BConv(channel, channel, 3, s=1, g=channel, act=nn.SiLU())
            # BConv(channel, channel, 5, s=1, g=channel, act=nn.SiLU())
        )
        self.dw2 = BConv(channel, channel, kernel_size, g=channel, act=nn.SiLU())

    def forward(self, x):
        out = self.compress(x)
        out = self.spatial(out)
        out = self.dw1(x) * out + self.dw2(x)
        return out

class FCM_Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.spatial_gate = DSM_SpatialGate(dim)

        self.final_conv = BConv(2*dim, 2*dim, 1, 1, g=2*dim, act=nn.SiLU())
        self.conv2 = BConv(dim, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        rgb, ir= x[0], x[1]
        x_rgb = self.spatial_gate(rgb)

        x_ir = self.conv2(ir)

        ir_out = self.spatial(x_rgb) * x_ir
        rgb_out = self.channel(x_ir) * x_rgb

        output = self.final_conv(torch.cat((rgb_out, ir_out), dim=1))

        return output  


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 5
        self.compress = ZPool()
        self.conv = BConv(2, 1, kernel_size, s=1)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class Channel_all(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            2*dim, dim, 3,
            1, 1, groups=dim
        )
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)

        return x6

class Spatial_all(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(2*dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)

        return x6

class Channel_all2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d( 
            3*dim, dim, 3,     
            1, 1, groups=dim
        )  # 反转output为2*
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)

        return x6

class Spatial_all2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3*dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)

        return x6

class Spatial_all_abl(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(int(2.5*dim), 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)

        return x6


class Channel_all2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d( 
            int(3*dim), dim, 3,     
            1, 1, groups=dim
        )  # 反转output为2*
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)

        return x6


import math

class Channel_all_abl(nn.Module):
    def __init__(self, dim):
        super().__init__()
        mid_groups = math.gcd(int(2.5*dim), dim)

        self.dwconv = nn.Conv2d( 
            int(2.5*dim), dim, 3,     
            1, 1, groups=mid_groups
        )  # 反转output为2*
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)

        return x6


class RGB_Gate(nn.Module):
    def __init__(self, in_size, local_size=5, gamma = 2, b = 1,local_weight=0.5):
        super(RGB_Gate, self).__init__()

        # ECA 计算方法
        self.local_size=local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)   # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)


        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)

    def forward(self, x):
        local_arv=self.local_arv_pool(x)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)

        y_local = self.conv_local(temp_local)
        # y_global = self.conv(temp_global)

        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b, c, self.local_size , self.local_size)

    
        att_local = y_local_transpose.sigmoid()
        att_all = F.adaptive_avg_pool2d(att_local, [m, n])

        x = x * att_all
        return x

class Fusion_nod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.spatial_gate = RGB_Gate(in_size=dim)

        self.conv2 = BConv(dim, dim, 1, 1)  # 3; 5
        
        self.spatial = Spatial_all(dim)
        self.channel = Channel_all(dim)

    def forward(self, x):
        rgb, ir= x[0], x[1]

        output = torch.cat((rgb, ir), dim=1)

        x_rgb = self.spatial_gate(rgb)  # 语义信息


        x_ir = self.conv2(ir)  # 位置信息


        ir_out = self.spatial(output) * x_ir
        rgb_out = self.channel(output) * x_rgb


        output_final = torch.cat((rgb_out, ir_out), dim=1)

        return output_final 


class Fusion_nod2(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.spatial_gate = RGB_Gate(in_size=dim)  # 2*
        self.conv2 = BConv(2*dim, 2*dim, 1, 1)  # -2*    # 3; 5

        
        self.spatial = Spatial_all2(dim)
        self.channel = Channel_all2(dim) # 反转output为2*

    def forward(self, x):
        rgb, ir= x[0], x[1]


        output = torch.cat((rgb, ir), dim=1)

        x_rgb = self.spatial_gate(rgb)  # 语义信息


        x_ir = self.conv2(ir)  # 位置信息2


        ir_out = self.spatial(output) * x_ir
        rgb_out = self.channel(output) * x_rgb


        output_final = torch.cat((rgb_out, ir_out), dim=1)

        return output_final 



class Spatial_all_abl2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)

        return x6

class Channel_all_abl2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        mid_groups = math.gcd(dim, 576)

        self.dwconv = nn.Conv2d( 
            dim, 576, 3,     
            1, 1, groups=mid_groups
        )  # 反转output为2*
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)

        return x6

class Fusion_nod_abl(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.spatial_gate = RGB_Gate(in_size=576)  # 2*
        self.conv2 = BConv(768, 768, 1, 1)  # -2*    # 3; 5

        self.spatial = Spatial_all_abl2(1344)
        self.channel = Channel_all_abl2(dim) # 反转output为2*

    def forward(self, x):
        rgb, ir= x[0], x[1]


        output = torch.cat((rgb, ir), dim=1)

        x_rgb = self.spatial_gate(rgb)  # 语义信息

        x_ir = self.conv2(ir) 

        ir_out = self.spatial(output) * x_ir
        rgb_out = self.channel(output) * x_rgb

        output_final = torch.cat((rgb_out, ir_out), dim=1)

        return output_final 



class Concat_Fusio_DSM(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1, c2=128):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

        self.module = Fusion_nod(c2) 

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return self.module(x)  
 
if __name__ == "__main__":
    x = [torch.randn(1, 64, 128, 128), torch.randn(1, 64, 128, 128)]
    # model = MutilScaleEdgeInformationEnhance(64, [1, 2, 4])
    # out = model(x)
    # print(out.shape)

    model = Concat_Fusio_DSM(c2=64)
    out = model(x)
    print(out.shape)