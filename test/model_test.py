import argparse
import os
import torch
import yaml

from models.yolo import Model
from torchstat import stat

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # yolov5s-voc/yolov5s-ghostconv-bifpn1-ca-voc
    parser.add_argument('--cfg', type=str,
                        default='../models/abnormal_driving/yolov5s_dd.yaml',
                        help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='../data/hyps/hyp.scratch-high.yaml',
                        help='hyperparameters path')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    with open(opt.hyp, encoding='utf-8', errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict 字典形式

    # 如果配置文件中有中文，打开时要加encoding = 'utf-8'参数
    with open(opt.cfg, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)  # model dict 取到配置文件中每条的信息

    nc = cfg['nc']  # 获取数据集的类别数
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    # input_img = torch.zeros(size=(1, 3, 1280, 1280))
    input_img = torch.zeros(size=(1, 3, 640, 640))
    input_img = input_img.to(device, non_blocking=True).float()
    print(f'the model of \'{opt.cfg}\' is :')
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    output = model(input_img)

    print(f'number of detect layers: {len(output)}')
    print('Detect head output: ')
    # print(f'P2/4: {output[len(output) - 4].shape}')
    print(f'P3/8: {output[len(output) - 3].shape}')
    print(f'P4/16: {output[len(output) - 2].shape}')
    print(f'P5/32: {output[len(output) - 1].shape}')

    '''
    3
    torch.Size([1, 3, 80, 80, 85])
    torch.Size([1, 3, 40, 40, 85])
    torch.Size([1, 3, 20, 20, 85])
    '''


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
