import torch

print("torch版本号: ", end="")
print(torch.__version__)

print("判断torch是否可用: ", end="")
print(torch.cuda.is_available())

print("gpu数量: ", end="")
print(torch.cuda.device_count())

print("gpu名字，设备索引默认从0开始: ", end="")
print(torch.cuda.get_device_name(0))
print("现在正在使用的GPU编号: ", end="")
print(torch.cuda.current_device())