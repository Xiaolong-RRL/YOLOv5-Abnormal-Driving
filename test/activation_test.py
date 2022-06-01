import torch
import torch.nn as nn
from matplotlib import pyplot as plt

relu = nn.ReLU()
leaky_relu = nn.LeakyReLU()
mish = nn.Mish()
silu = nn.SiLU()

x = torch.linspace(-10, 10, 1000)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_mish = mish(x)
y_silu = silu(x)

plt.plot(x, y_relu, 'g-')
plt.plot(x, y_leaky_relu, 'y-')
plt.plot(x, y_mish, 'b-')
plt.plot(x, y_silu, 'r-')
plt.grid()
plt.show()
