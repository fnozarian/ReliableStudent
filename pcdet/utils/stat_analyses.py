from pcdet.ops.iou3d_nms import iou3d_nms_utils
import torch


def get_false_positive(gt_boxes, predictions, class_names, iou_thresh=0.1):
    fp = {}
    k = gt_boxes.shape[0]
    while gt_boxes[k-1, 7] == 0:
        k = k - 1
    gt_boxes_new = gt_boxes[:k, :]
    for i, class_name in enumerate(class_names):
        indices = predictions[:, 7] == i + 1
        ps_labels_class = predictions[indices, :]
        indices = gt_boxes_new[:, 7] == i + 1
        gt_boxes_class = gt_boxes_new[indices, :]
        if 0 == ps_labels_class.__len__():
            fp[class_name] = 0.
            return fp
        if 0 == gt_boxes_class.__len__():
            fp[class_name] = float(ps_labels_class.shape[0])
            return fp

        real_ps_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
            ps_labels_class[:, :7], gt_boxes_class[:, :7])
        if 0 < real_ps_by_gt_overlap.__len__():
            iou_max, asgn = real_ps_by_gt_overlap.max(dim=1)
            false_positive = torch.tensor((iou_max < iou_thresh) & (0 < iou_max), dtype=float).sum()
            fp[class_name] = false_positive

    return fp


# based on st3d "gt boxes that missed by tp boxes are fn boxes", there is a problem in st3d
def get_false_negatives(gt_boxes, predictions, class_names, iou_thresh=0.1):
    fn = {}
    k = gt_boxes.shape[0]
    while gt_boxes[k-1, 7] == 0:
        k = k - 1
    gt_boxes = gt_boxes[:k, :]
    for i, class_name in enumerate(class_names):
        indices = predictions[:, 7] == i + 1
        ps_labels_class = predictions[indices, :]
        indices = gt_boxes[:, 7] == i + 1
        gt_boxes_class = gt_boxes[indices, :]
        if ps_labels_class.__len__() == 0:
            fn[class_name] = float(gt_boxes_class.shape[0])
            return fn
        if gt_boxes_class.__len__() == 0:
            fn[class_name] = -1
            return fn

        real_gt_by_ps_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
            gt_boxes_class[:, :7], ps_labels_class[:, :7])
        if 0 < real_gt_by_ps_overlap.__len__():
            iou_max, asgn = real_gt_by_ps_overlap.max(dim=1)
            miss = torch.tensor(iou_max < iou_thresh, dtype=float).sum()
            fn[class_name] = miss

    return fn


def get_true_positive(gt_boxes, predictions, class_names, iou_thresh=0.1):
    tp = {}
    k = gt_boxes.shape[0]
    while gt_boxes[k-1, 7] == 0:
        k = k - 1
    gt_boxes = gt_boxes[:k, :]
    for i, class_name in enumerate(class_names):
        indices = predictions[:, 7] == i + 1
        ps_labels_class = predictions[indices, :]
        indices = gt_boxes[:, 7] == i + 1
        gt_boxes_class = gt_boxes[indices, :]
        if 0 == gt_boxes_class.__len__():
            tp[class_name] = 0.
            return tp
        if 0 == ps_labels_class.__len__():
            tp[class_name] = -1.
            return tp

        real_ps_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
            ps_labels_class[:, :7], gt_boxes_class[:, :7])
        if 0 < real_ps_by_gt_overlap.__len__():
            iou_max, asgn = real_ps_by_gt_overlap.max(dim=1)
            hit_real_gt = torch.tensor(iou_max >= iou_thresh, dtype=float).sum()
            tp[class_name] = hit_real_gt

    return tp


def pseudo_labels_vs_gt_precision(gt_boxes, ps_labels, class_names, iou_thresh=0.1):
    precision = {}
    fps = get_false_positive(gt_boxes, ps_labels, class_names, iou_thresh=iou_thresh)
    tps = get_true_positive(gt_boxes, ps_labels, class_names, iou_thresh=iou_thresh)

    for i, class_name in enumerate(class_names):
        precision[class_name] = -1.
        if class_name in tps and class_name in fps:
            if tps[class_name] != 0 or fps[class_name] != 0:
                if tps[class_name] != -1 and fps[class_name] != -1:
                    precision[class_name] = tps[class_name] / (tps[class_name] + fps[class_name])

    return precision


if __name__ == '__main__':
    ps_path = 'salam'

    print(ps_path)