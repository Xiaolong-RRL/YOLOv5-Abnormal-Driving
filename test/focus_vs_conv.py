import time

import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3,
                              out_channels=32,
                              kernel_size=6,
                              stride=2,
                              padding=2)
        self.bn = nn.BatchNorm2d(32)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3 * 4,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=3 // 2),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # 假设x = [1,2,3,4,5,6,7,8,9] x[::2] = [1,3,5,7,9] 间隔2个取样
        # x[1::2] = [2, 4, 6, 8] 从第二个数据开始，间隔2个取样
        return self.conv(torch.cat([x[..., ::2, ::2],
                                    x[..., 1::2, ::2],
                                    x[..., ::2, 1::2],
                                    x[..., 1::2, 1::2]], 1))


if __name__ == '__main__':
    input_data = torch.rand(1, 3, 640, 640)

    conv = Conv()
    focus = Focus()

    output1 = conv(input_data)
    output2 = focus(input_data)

    print(f"output1 size: {output1.size()}")
    print(f"output2 size: {output2.size()}")
    print(torch.equal(output1, output2))

    # 速度对比
    t_start = time.time()
    for _ in range(300):
        conv(input_data)
    print(f"6x6 conv time: {time.time() - t_start}")

    t_start = time.time()
    for _ in range(300):
        focus(input_data)
    print(f"focus time: {time.time() - t_start}")

'''
output1 size: torch.Size([1, 32, 320, 320])
output2 size: torch.Size([1, 32, 320, 320])
False
6x6 conv time: 2.4613001346588135
focus time: 2.8948004245758057
'''
