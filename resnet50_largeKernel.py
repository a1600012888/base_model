import torch
import torch.nn as nn

import torch.nn.functional as F



class conv_bn_relu(nn.Module):

    def __init__(self, in_chn, out_chn, kernel_size = 3, stride = 1, padding = 1, dilation = 1,
                 has_bias = False, has_bn = rue, has_relu = True):

        super(conv_bn_relu, self).__init__()

        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size, stride, padding, dilation, has_bias)

        if has_bn:
            self.bn = nn.BatchNorm2d(out_chn)

        if has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class Botteneck(nn.Module):

    def __init__(self, in_chn, mid_chn, out_chn, stride, dilation = 1, has_proj = False):

        super(Botteneck, self).__init__()

        if has_proj:
            self.proj = conv_bn_relu(in_chn, out_chn, kernel_size = 1, stride = stride,
                                     padding = 0, dilation = dilation)

        self.conv1 = conv_bn_relu(in_chn, mid_chn, 1, stride, padding = 0)

        self.conv2 = conv_bn_relu(mid_chn, mid_chn, 3, 1, padding = dilation)

        self.conv3 = conv_bn_relu(mid_chn, out_chn, 1, 1, padding = 0, has_relu = False)

    def forward(self, x):

        proj = x
        if self.proj:
            proj = self.proj(proj)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = proj + x

        x = F.relu(x)

        return x

def make_stage(in_chn, mid_chn, out_chn, num_bottles, enable_stride = True):

    blocks = []

    blocks.append(Botteneck(in_chn, mid_chn, out_chn, stride = 2 if enable_stride else 1, has_proj=True))

    for i in range(1, num_bottles):
        blocks.append(out_chn, mid_chn, out_chn, stride = 1)

    return nn.Sequential(*blocks)

class resnet50_largeKernel(nn.Module):

    def __init__(self):

        self.conv1 = conv_bn_relu(in_chn=3, out_chn=64, kernel_size=3, stride = 2)

        self.conv2 = conv_bn_relu(in_chn=64, out_chn=64)

        self.conv3 = conv_bn_relu(in_chn=64, out_chn=128)

        self.Maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_bottles = [3, 4, 6, 3]
        mild_outputs = [64, 128, 256, 512]
        outputs = [256, 512, 1024, 2048]
        enable_stride = [False, True, True, True]
        dilation = 1

        self.stage1 = make_stage(128, mild_outputs[0], outputs[0], enable_stride[0])
        self.stage2 = make_stage(outputs[0], mild_outputs[1], outputs[1], enable_stride[1])
        self.stage3 = make_stage(outputs[1], mild_outputs[2], outputs[2], enable_stride[2])
        self.stage4 = make_stage(outputs[2], mild_outputs[3], outputs[3], enable_stride[3])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.Maxpool(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        return x1, x2, x3, x4



class GCN(nn.Module):

    def __init__(self, in_chn, out_chn = 21, large_kernel = 21):

        self.conv_l1= nn.Conv2d()