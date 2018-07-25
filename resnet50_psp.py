import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class conv_bn_relu(nn.Module):

    def __init__(self, in_chn, out_chn, kernel_size = 3, stride = 1, padding = 1, dilation = 1,
                 has_bias = False, has_bn = True, has_relu = True):

        super(conv_bn_relu, self).__init__()

        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size, stride, padding, dilation, bias = has_bias)

        self.bn = None
        self.relu = None
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

class resnet50_psp(nn.Module):

    def __init__(self, pretrained = True):
        super(resnet50_psp, self).__init__()
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

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.Maxpool(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        return x4, x3


class psp_branch(nn.Module):

    def __init__(self, in_chn, out_chn, size = 1):

        super(psp_branch, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((size, size))

        self.conv_reduce = conv_bn_relu(in_chn, out_chn, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        #print(x.size())
        x = self.conv_reduce(x)

        x = F.upsample(x, size = (h, w), mode = 'bilinear')

        return x

class pspModule(nn.Module):

    def __init__(self, in_chn):

        super(pspModule, self).__init__()

        sizes = (1, 2, 3, 6)
        mid_chns = 512

        self.branch1 = psp_branch(in_chn, mid_chns, size = sizes[0])
        self.branch2 = psp_branch(in_chn, mid_chns, size=sizes[1])
        self.branch3 = psp_branch(in_chn, mid_chns, size=sizes[2])
        self.branch4 = psp_branch(in_chn, mid_chns, size=sizes[3])

    def forward(self, x):

        psp_b1 = self.branch1(x)
        psp_b2 = self.branch2(x)
        psp_b3 = self.branch3(x)
        psp_b4 = self.branch4(x)

        feature = torch.cat((x, psp_b1, psp_b2, psp_b3, psp_b4), dim = 1)

        return feature

class predictBranch(nn.Module):
    '''
    The final prediction branch after psp pool in psp
    '''

    def __init__(self, in_chn, mid_chn, nb_class, size):
        '''

        :param in_chn:
        :param mid_chn:
        :param nb_class: channels of the final prediction
        :param size: size of the final output: (h, w)
        '''
        super(predictBranch, self).__init__()
        self.conv1 = conv_bn_relu(in_chn, mid_chn)

        self.drop = nn.Dropout2d(p = 1, inplace=False)

        self.conv2 = conv_bn_relu(mid_chn, nb_class, 1, 1, 0, has_bias=True, has_relu=False, has_bn=False)

        self.size = size
    def forward(self, x):

        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)

        x = F.upsample(x, self.size, mode = 'bilinear')

        return x
class PSP(nn.Module):
    '''
    backbone as resnet-50
    '''

    def __init__(self, nb_class, size = (480, 480)):

        super(PSP, self).__init__()

        self.backbone = resnet50_psp()

        self.psp = pspModule(in_chn=2048)

        self.score = predictBranch(4096, 512, nb_class, size = size)

        self.aux = predictBranch(1024, 1024, nb_class, size = size)

    def forward(self, x):

        x4, x3 = self.backbone(x)

        psp_out = self.psp(x4)

        pred = self.score(psp_out)

        aux_pred = self.aux(x3)

        return pred, aux_pred


def test():

    net = PSP(3, size = (512, 256))
    inp = torch.randn(2, 3, 512, 256)

    pred, x = net(inp)

    print(pred.size())


def test_res():
    net = resnet50_psp()

    inp = torch.randn(2, 3, 512, 256)

    s4, s3 = net(inp)

    print(s4.size())
    print(s3.size())
if __name__ == '__main__':
    test()
    #test_res()
