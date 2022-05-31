import os
import random

'''对图片数据集进行随机分类
以8: 1: 1的比例分为训练数据集，验证数据集和测试数据集
运行后在ImageSets文件夹中会出现四个文件
'''

# ROOT = '../autodl-tmp/abnormal_driving/'
ROOT = '../datasets/bishe/abnormal_driving/'
train_percent = 0.9
xmlfilepath = ROOT + 'Annotations'
txtsavepath = ROOT + 'ImageSets'
# 获取该路径下所有文件的名称，存放在list中
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tr = int(num * train_percent)
train = random.sample(list, tr)

ftrain = open(ROOT + 'ImageSets/train.txt', 'w')
fval = open(ROOT + 'ImageSets/val.txt', 'w')

for i in list:
    # 获取文件名称中.xml之前的序号
    name = total_xml[i][:-4] + '\n'
    if i in train:
        ftrain.write(name)
    else:
        fval.write(name)

ftrain.close()
fval.close()
