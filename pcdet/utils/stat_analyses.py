from pcdet.ops.iou3d_nms import iou3d_nms_utils
import torch

def get_miss_rate(gt_boxes, predictions, threshold=0.5):
    gt_labels = gt_boxes[:, 7]
    pred_labels = predictions[:, 7]
    real_gt_by_ps_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
        gt_boxes[:, :7], predictions[:, :7])
    iou_max, asgn = real_gt_by_ps_overlap.max(dim=1)

    miss_real_gt = torch.tensor((iou_max + asgn) == 0, dtype=float).sum()
    miss_rate = miss_real_gt.unsqueeze(dim=0)
    return miss_rate