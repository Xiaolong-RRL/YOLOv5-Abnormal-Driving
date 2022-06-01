'''
stat
可以用来计算pytorch构建的网络的参数，空间大小，MAdd，FLOPs等指标，简单好用。
比如：我想知道alexnet的网络的一些参数
'''
from torchstat import stat
import torchvision.models as models

model = models.alexnet()
stat(model, (3, 224, 224))
