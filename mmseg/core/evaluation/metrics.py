# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import mmcv
import numpy as np
import torch


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def cal_metrics(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                label_map=dict(),
                reduce_zero_label=False):
    IoU_sum = 0.
    f_beta_sum = 0.
    beta = 1
    mse_sum = 0.
    for result, gt_seg_map in zip(results, gt_seg_maps):
        if len(result) == 2:
            seg_pred = result[0][0]
            if result[1][0].shape[0] == 2:
                seg_logit = seg_pred
            else:
                seg_logit = result[1][0]
        else:
            seg_pred = result
            seg_logit = result
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                seg_pred, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        iou_single = area_intersect[1] / area_union[1]
        IoU_sum += iou_single
        precision = area_intersect[1] / max(1, area_pred_label[1])
        recall = area_intersect[1] / max(1, area_label[1])
        f1_single = (1 + beta ** 2) * precision * recall / max((beta ** 2 * precision + recall), 0.001)
        f_beta_sum += f1_single
        prob_mse_single = np.mean((gt_seg_map - seg_logit) ** 2)
        mse_sum += prob_mse_single

    miou_by_image = IoU_sum / len(results)
    mf_beta_by_image = f_beta_sum / len(results)
    mse_by_image = mse_sum / len(results)
    
    return  miou_by_image, mf_beta_by_image, mse_by_image


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):

    miou_by_image, mf_beta_by_image, mmse_by_image = cal_metrics(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)

    ret_metrics = total_metrics(miou_by_image, mf_beta_by_image, 
                                mmse_by_image,metrics, nan_to_num) 
    
    return ret_metrics


def total_metrics(miou_by_image,
                  mf_beta_by_image,
                  mse_by_image,
                  metrics=['mIoU'],
                  nan_to_num=None,
                  ):
    
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'mMse', 'SmokeSeg']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    ret_metrics = OrderedDict()
    for metric in metrics:
        if metric == 'SmokeSeg':
            ret_metrics['mf_beta'] = torch.tensor([mf_beta_by_image, mf_beta_by_image], dtype=torch.float64)
            ret_metrics['mIoU'] = torch.tensor([miou_by_image, miou_by_image], dtype=torch.float64)
            ret_metrics['mMse'] = torch.tensor([mse_by_image, mse_by_image], dtype=torch.float64)

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics

