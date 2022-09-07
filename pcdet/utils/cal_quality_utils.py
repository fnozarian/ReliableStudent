import torch
import math
import numpy as np
from pcdet.ops.iou3d_nms import iou3d_nms_utils

# Adopted from https://github.com/shashankag14/OpenPCDet/blob/8be4f979a916e3aba056d5a6106d3e52e819f6a8/pcdet/utils/cal_quality_utils.py
# with modification

"""Computes and stores the average and current value"""


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        assert np.isscalar(val)

        if not np.isnan(val):
            assert val >= 0, "Update value should be non negative"
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


'''Computes quality metrics - tp, fp, fn, precision, recall, assignment error, cls error and 
    num of rejections made due to pseudo label filtering
    
    Note : Adapted from st3d cal_quality_utils.py and modified'''


class Metrics(object):
    def __init__(self):
        self.metrics = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(),
                        'assignment_err': AverageMeter(), 'cls_err': AverageMeter(),
                        'precision': AverageMeter(), 'recall': AverageMeter(), 'rej_pseudo_lab': AverageMeter()}

    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()

    def get_true_pos(self, mask):
        return mask.sum().item()

    def get_false_pos(self, total_preds, tp):
        return total_preds - tp

    def get_false_neg(self, total_gts, tp):
        return total_gts - tp

    def get_precision_recall(self, tp, fp, fn):
        assert tp > 0, "True positives are zero"
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return precision, recall

    def get_cls_err(self, pseudo_labels, gt_labels, iou_match_ind, class_mask=None):
        # find correct matches b/w PLs and GTs 
        correct_matches = (gt_labels.gather(dim=0, index=iou_match_ind) == pseudo_labels).float()
        # if computing error class-wise, find correct matches class wise
        if class_mask != None:
            assert class_mask.shape == correct_matches.shape
            correct_matches = correct_matches[class_mask]
        cls_err = (1 - correct_matches.mean()).item()
        return cls_err

    @staticmethod
    def cor_angle_range(angle):
        """ correct angle range to [-pi, pi]
        Args:
            angle:
        Returns:
        """
        gt_pi_mask = angle > np.pi
        lt_minus_pi_mask = angle < - np.pi
        angle[gt_pi_mask] = angle[gt_pi_mask] - 2 * np.pi
        angle[lt_minus_pi_mask] = angle[lt_minus_pi_mask] + 2 * np.pi

        return angle

    '''Compute orientation/angle difference based on L1 distance'''

    def cal_angle_diff(self, angle1, angle2):
        # angle is from x to y, anti-clockwise
        angle1 = self.cor_angle_range(angle1)
        angle2 = self.cor_angle_range(angle2)

        diff = torch.abs(angle1 - angle2)
        gt_pi_mask = diff > math.pi
        diff[gt_pi_mask] = 2 * math.pi - diff[gt_pi_mask]

        return diff

    # TODO(farzad) assignment error is for predictions with IoU close and bellow the FG threshold
    #  which are considered as background or another object.
    def get_assignment_err(self, tp_boxes, gt_boxes, tp):
        assert tp_boxes.shape[0] == gt_boxes.shape[0]
        assert tp > 0, "True positives are zero"
        # L2 distance xy only
        center_distance = torch.norm(tp_boxes[:, :2] - gt_boxes[:, :2], dim=1)
        trans_err = center_distance.sum().item()

        # Angle difference
        angle_diff = self.cal_angle_diff(tp_boxes[:, 6], gt_boxes[:, 6])
        assert angle_diff.sum() >= 0
        orient_err = angle_diff.sum().item()

        # Scale difference
        aligned_tp_boxes = tp_boxes.detach().clone()
        # shift their center together
        aligned_tp_boxes[:, 0:3] = gt_boxes[:, 0:3]
        # align their angle
        aligned_tp_boxes[:, 6] = gt_boxes[:, 6]
        iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(aligned_tp_boxes[:, 0:7], gt_boxes[:, 0:7])
        max_ious, _ = torch.max(iou_matrix, dim=1)
        scale_err = (1 - max_ious).sum().item()

        return trans_err / tp + orient_err / tp + scale_err / tp


'''Hold record for class-wise metrics'''
class ClassWiseMetric(object):
    def __init__(self, class_names):
        self.class_names = class_names
        self.class_metrics = {cls: Metrics() for cls in class_names}

    '''reset the records at the beginning of each epoch'''
    def reset(self):
        for key in self.class_metrics.keys():
            self.class_metrics[key].reset()

    def get_metrics_of(self, cls):
        return self.class_metrics[cls]

    '''update metrics - tp, fp, fn, assignment err and cls err class wise'''
    def update_metrics_of_all_classes(self, pseudo_boxes, gt_boxes, iou_max, fg_thresh, asgn, cls_pseudo):
        # Create mask for preds satifying the FG threshold and compute the metrics on that mask
        tp_mask = iou_max >= fg_thresh
        gt_labels = gt_boxes[:, 7]
        # Used for FNs : Count total number of gt labels per class 
        num_gt_labels_per_class = torch.bincount(gt_labels.type(torch.int64), minlength=4)

        for cls, metrics in self.class_metrics.items():
            # find class wise mask
            cls_id = self.class_names.index(cls) + 1
            class_mask = cls_pseudo == cls_id
            class_tp_mask = tp_mask & class_mask
            # tp, fp, fn, cls err
            tp = metrics.get_true_pos(class_tp_mask)
            fp = metrics.get_false_pos(iou_max[class_mask].shape[0], tp)
            fn = metrics.get_false_neg(num_gt_labels_per_class[cls_id].item(), tp)
            cls_err = metrics.get_cls_err(cls_pseudo, gt_boxes[:, 7], asgn, class_mask)

            # compute assignment error, precision and recall ONLY if there are any TPs
            if tp:
                # get tp boxes and their corresponding gt boxes
                tp_pseudo_boxes_per_class = pseudo_boxes[class_tp_mask]
                tp_gt_boxes_per_class = gt_boxes[asgn, :][class_tp_mask]
                assignment_err = metrics.get_assignment_err(tp_pseudo_boxes_per_class, tp_gt_boxes_per_class, tp)
                precision, recall = metrics.get_precision_recall(tp, fp, fn)

            metrics.metrics['tp'].update(tp)
            metrics.metrics['fp'].update(fp)
            metrics.metrics['fn'].update(fn)
            metrics.metrics['cls_err'].update(cls_err)
            if tp:
                metrics.metrics['assignment_err'].update(assignment_err)
                metrics.metrics['precision'].update(precision)
                metrics.metrics['recall'].update(recall)


class MetricRegistry(object):
    def __init__(self, class_name):
        self._tag_metrics = {}
        self.class_name = class_name

    def get(self, tag=None):
        if tag is None:
            tag = 'default'
        if tag in self._tag_metrics.keys():
            metric = self._tag_metrics[tag]
        else:
            metric = ClassWiseMetric(self.class_name)
            self._tag_metrics[tag] = metric
        return metric

    def tags(self):
        return self._tag_metrics.keys()