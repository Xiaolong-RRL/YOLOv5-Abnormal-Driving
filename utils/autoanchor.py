# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
AutoAnchor utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import LOGGER, colorstr, emojis

PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary
    # m: 模型最后一层的输出(Detect层)
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()

    # dataset.shapes.max(1, keepdims=True) = 每张图片的较长边
    # shapes: 将数据集图片的最长边缩放到img_size 较小边相应缩放 得到新的所有数据集图片的宽高 [8069, 2]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)

    # 产生[0.9, 1.1]之间的随机尺度 [8069, 1]
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale

    # s: 数据集所有图片宽高缩放到0-imgsz后，再乘上随机尺度值
    # l: 数据集的labels标签[class_id, x, y, w, h] 其中坐标信息都是归一化后的数值
    # wh: [24638]，就是基于原图，且做了随机尺度缩放(shapes * scale)后的图像GT框的宽高
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        """用在check_anchors函数中  compute metric
        根据数据集中所有图片的wh与当前所有anchors k计算 bpr(best possible recall) 和 aat(anchors above threshold)
        :params k: anchors [9, 2]  wh: [N, 2]
        :return bpr: best possible recall 最多能被召回(通过thr)的gt框数量 / 所有gt框数量   小于0.98 才会用k-means计算anchor
        :return aat: anchors above threshold 每个target平均有多少个anchors
         """

        # wh[:, None]: [24638, 2]->[24638, 1, 2]
        # k[None]: [9, 2]->[1, 9, 2]
        # r: [24638, 9, 2] GT框的高h宽w与anchor的高h_a宽w_a的比值，即h/h_a, w/w_a  [8069, 9, 2]  有可能大于1，也可能小于等于1
        r = wh[:, None] / k[None]

        # x: [24638, 9] 高宽比和宽高比的最小值 无论r大于1，还是小于等于1最后统一结果都要小于1
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric

        # best: [24638] 为每个gt框选择匹配所有anchors宽高比例值最好的那一个比值
        best = x.max(1)[0]  # best_x

        # (x > 1. / thr): x中大于1./thr的赋值为1，小于等于的赋值为0 [24638, 9]
        # .sum(1): 是将一行中的9个值（0 or 1）求和 [24638]
        # .mean(): 将这24638个数值求平均，得到的就是每个target平均有多少满足阈值条件的个anchors
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold

        # 最多能被召回(通过thr)的gt框数量 / 所有gt框数量   小于0.98 才会用k-means计算anchor
        # 这里也是1表示通过thr，0表示未通过，然后取平均
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    # m.anchor_grid: 存储着当前默认的anchors的宽高，而m.anchor则存储着归一化后的anchors的宽高
    # 获得所有anchors的宽高并resize到[N, 2]
    anchors = m.anchors.clone() * m.stride.to(m.anchors.device).view(-1, 1, 1)  # current anchors

    # 计算出数据集中所有图片的wh与当前所有anchors的bpr和aat
    # bpr: best possible recall  最多能被召回(通过thr)的gt框数量 / 所有gt框数量  小于0.98 才会用k-means计算anchor
    # aat: anchors past thr  通过阈值的anchor个数
    bpr, aat = metric(anchors.cpu().view(-1, 2))

    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(emojis(f'{s}Current anchors are a good fit to dataset ✅'))
    else:
        LOGGER.info(emojis(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...'))
        na = m.anchors.numel() // 2  # number of anchors
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            LOGGER.info(f'{PREFIX}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            LOGGER.info(f'{PREFIX}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            LOGGER.info(f'{PREFIX}Original anchors better than new anchors. Proceeding with original anchors.')


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for i, x in enumerate(k):
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans calculation
    LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
    s = wh.std(0)  # sigmas for whitening
    k = kmeans(wh / s, n, iter=30)[0] * s  # points
    if len(k) != n:  # kmeans may return fewer points than requested if wh is insufficient or too similar
        LOGGER.warning(f'{PREFIX}WARNING: scipy.cluster.vq.kmeans returned only {len(k)} of {n} requested points')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'{PREFIX}Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k)
