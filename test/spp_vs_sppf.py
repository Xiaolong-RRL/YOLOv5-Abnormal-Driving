import torch.nn as nn
import torch
import time


class SPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

    def forward(self, x):
        p1 = self.maxpool1(x)
        p2 = self.maxpool2(x)
        p3 = self.maxpool3(x)
        return torch.cat([x, p1, p2, p3], dim=1)


class SPPF(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)

    def forward(self, x):
        p1 = self.maxpool(x)
        p2 = self.maxpool(p1)
        p3 = self.maxpool(p2)
        return torch.cat([x, p1, p2, p3], dim=1)


if __name__ == '__main__':
    input_data = torch.rand(8, 32, 64, 64)
    spp = SPP()
    sppf = SPPF()
    output1 = spp(input_data)
    output2 = sppf(input_data)

    print(torch.equal(output1, output2))

    t_start = time.time()
    for _ in range(100):
        spp(input_data)
    print(f"spp time: {time.time() - t_start}")

    t_start = time.time()
    for _ in range(100):
        sppf(input_data)
    print(f"sppf time: {time.time() - t_start}")

'''
True
spp time: 4.364669561386108
sppf time: 1.5898349285125732
'''
