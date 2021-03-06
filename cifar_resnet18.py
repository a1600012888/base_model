import torch
import numpy as np
import torch.nn  as nn
import torch.nn.functional as F
class resnet18_3x3_3x3(nn.Module):


    def __init__(self, inp_chn, out_chn, stride = 1, has_proj = False):

        super(resnet18_3x3_3x3, self).__init__()
        self.conv1 = nn.Conv2d(inp_chn, out_chn, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_chn)
        self.conv2 = nn.Conv2d(out_chn, out_chn, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_chn)
        self.proj = None
        if has_proj:
            self.proj = nn.Conv2d(inp_chn, out_chn, kernel_size = 1, stride = stride, padding = 0)

    def forward(self, inp):

        x = inp

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        proj = inp
        if self.proj != None:
            proj = self.proj(proj)

        x = x + proj
        x = F.relu(x)
        return x

def make_resnet18_block(depth, inp_chn, out_chn, stride = 2):
    blocks = []
    blocks.append(resnet18_3x3_3x3(inp_chn, out_chn, stride, True))
    for i in range(depth - 1):
        blocks.append(resnet18_3x3_3x3(out_chn, out_chn, 1, False))

    return nn.Sequential(*blocks)

class cifar_resnet18(nn.Module):

    def __init__(self, num_class = 10, expansion = 2):
        '''
        expansion: standard resnet-18 has channels as [16, 32, 64]
                   expand the channel in standard resnet-18
        '''

        super(cifar_resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1)

        self.bn1 = nn.BatchNorm2d(16)

        self.block1 = make_resnet18_block(2, 16, 16 * expansion, stride = 1)
        self.block2 = make_resnet18_block(3, 16 * expansion, 32 * expansion, stride = 2)
        self.block3 = make_resnet18_block(4, 32 * expansion, 64 * expansion, stride = 2)

        self.global_avg_pool = nn.AvgPool2d(kernel_size = 8)

        self.fc = nn.Linear(64 * expansion, num_class)

        self.kaiming_init()
    def forward(self, inp):

        x = self.conv1(inp)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)


def test():

    inp = np.random.randn(1, 3, 32, 32)
    inp = torch.tensor(inp, dtype = torch.float32).cuda()

    net = cifar_resnet18().cuda()
    print(net)
    pred = net(inp)
    print(net.conv1.data)
    print(pred)


if __name__ == '__main__':

    test()
