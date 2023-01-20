# arguments example:
# --pred_infos
# <OpenPCDet_HOME>/output/cfgs/kitti_models/pv_rcnn_ssl/enabled_st_all_bs8_dist4_split_1_2_trial3_169035d/eval/eval_with_train/epoch_60/val/result.pkl
# --gt_infos
# <OpenPCDet_HOME>/data/kitti/kitti_infos_val.pkl

import argparse
import pickle

from torchmetrics import Metric
import torch
import numpy as np
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import math
from pcdet.config import cfg
from matplotlib import pyplot as plt
import torch.nn.functional as F
# TODO(farzad): Pass only scores and labels?
#               Calculate overlap inside update or compute?
#               Change the states to TP, FP, FN, etc?
#               Calculate incrementally based on summarized value?


class PredQualityMetrics(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_state_interval = kwargs.get('reset_state_interval', 64)
        self.tag = kwargs.get('tag', None)
        self.dataset = kwargs.get('dataset', None)
        self.config = kwargs.get('config', None)
        # We use _fg, _bg and _uc if a pred is fg, bg or uc wrt ground truth, respectively.
        # We use _tp if a pred is fg wrt gt and fg wrt pl.
        # We use _fn if a pred is fg wrt gt and not fg wrt pl.
        # We use _fp if a pred is not fg wrt gt and fg wrt pl.
        self.metrics_name = ["pred_ious", "pred_fgs", "sem_score_fgs", "sem_score_bgs", "score_fgs", "score_bgs",
                             "target_score_bg", "num_pred_boxes", "num_gt_boxes", "pred_weight_fg", "pred_weight_bg",
                             "pred_ucs", "pred_ious_ucs", "score_ucs", "sem_score_ucs", "target_score_uc",
                             "pred_weight_uc", "pred_fn_rate", "pred_tp_rate", "pred_fp_ratio", "pred_ious_wrt_pl_fg",
                             "pred_ious_wrt_pl_fn", "pred_ious_wrt_pl_fp", "pred_ious_wrt_pl_tp", "score_fgs_tp",
                             "score_fgs_fn", "score_fgs_fp", "target_score_fn", "target_score_tp", "target_score_fp",
                             "pred_weight_fn", "pred_weight_tp", "pred_weight_fp"]
        self.min_overlaps = np.array([0.7, 0.5, 0.5, 0.7, 0.5, 0.7])
        self.class_agnostic_fg_thresh = 0.7
        for metric_name in self.metrics_name:
            self.add_state(metric_name, default=[], dist_reduce_fx='cat')

    def update(self, preds: [torch.Tensor], ground_truths: [torch.Tensor], pred_scores: [torch.Tensor],
               rois=None, roi_scores=None, targets=None, target_scores=None, pred_weights=None,
               pseudo_labels=None, pseudo_label_scores=None, pred_iou_wrt_pl=None) -> None:
        assert isinstance(preds, list) and isinstance(ground_truths, list) and isinstance(pred_scores, list)
        assert all([pred.dim() == 2 for pred in preds]) and all([pred.dim() == 2 for pred in ground_truths]) and all([pred.dim() == 1 for pred in pred_scores])
        assert all([pred.shape[-1] == 8 for pred in preds]) and all([gt.shape[-1] == 8 for gt in ground_truths])
        if roi_scores is not None:
            assert len(pred_scores) == len(roi_scores)

        roi_scores = [score.clone().detach() for score in roi_scores] if roi_scores is not None else None
        preds = [pred_box.clone().detach() for pred_box in preds]
        pred_scores = [ps_score.clone().detach() for ps_score in pred_scores]
        pred_iou_wrt_pl = [iou.clone().detach() for iou in pred_iou_wrt_pl] if pred_iou_wrt_pl is not None else None
        target_scores = [target_score.clone().detach() for target_score in target_scores] if target_scores is not None else None
        ground_truths = [gt_box.clone().detach() for gt_box in ground_truths]
        pseudo_labels = [pl_box.clone().detach() for pl_box in pseudo_labels] if pseudo_labels is not None else None
        pred_weights = [pred_weight.clone().detach() for pred_weight in pred_weights] if pred_weights is not None else None

        sample_tensor = preds[0] if len(preds) else ground_truths[0]
        num_classes = len(self.dataset.class_names)
        for i in range(len(preds)):
            valid_preds_mask = torch.logical_not(torch.all(preds[i] == 0, dim=-1))
            valid_pred_boxes = preds[i][valid_preds_mask]

            valid_pred_scores = pred_scores[i][valid_preds_mask.nonzero().view(-1)]
            valid_roi_scores = roi_scores[i][valid_preds_mask.nonzero().view(-1)] if roi_scores else None
            valid_target_scores = target_scores[i][valid_preds_mask.nonzero().view(-1)] if target_scores else None
            valid_pred_weights = pred_weights[i][valid_preds_mask.nonzero().view(-1)] if pred_weights else None
            valid_pred_iou_wrt_pl = pred_iou_wrt_pl[i][valid_preds_mask.nonzero().view(-1)].squeeze() if pred_iou_wrt_pl else None
            valid_gts_mask = torch.logical_not(torch.all(ground_truths[i] == 0, dim=-1))
            valid_gt_boxes = ground_truths[i][valid_gts_mask]
            if pseudo_labels is not None:
                valid_pl_mask = torch.logical_not(torch.all(pseudo_labels[i] == 0, dim=-1))
                valid_pl_boxes = pseudo_labels[i][valid_pl_mask] if pseudo_labels else None
                valid_pl_boxes[:, -1] -= 1

            # Starting class indices from zero
            valid_pred_boxes[:, -1] -= 1
            valid_gt_boxes[:, -1] -= 1

            # Adding predicted scores as the last column
            valid_pred_boxes = torch.cat([valid_pred_boxes, valid_pred_scores.unsqueeze(dim=-1)], dim=-1)

            pred_labels = valid_pred_boxes[:, -2]

            num_gts = valid_gts_mask.sum()
            num_preds = valid_preds_mask.sum()

            classwise_metrics = {}
            for metric_name in self.metrics_name:
                classwise_metrics[metric_name] = sample_tensor.new_zeros(num_classes + 1).fill_(float('nan'))

            for cind in range(num_classes):
                pred_cls_mask = pred_labels == cind
                gt_cls_mask = valid_gt_boxes[:, -1] == cind
                classwise_metrics['num_pred_boxes'][cind] = pred_cls_mask.sum()
                classwise_metrics['num_gt_boxes'][cind] = gt_cls_mask.sum()

                if num_gts > 0 and num_preds > 0:
                    overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_pred_boxes[:, 0:7], valid_gt_boxes[:, 0:7])
                    preds_iou_max, assigned_gt_inds = overlap.max(dim=1)

                    assigned_gt_cls_mask = valid_gt_boxes[assigned_gt_inds, -1] == cind

                    cc_mask = (pred_cls_mask & assigned_gt_cls_mask)  # correctly classified mask
                    mc_mask = (pred_cls_mask & (~assigned_gt_cls_mask)) | ((~pred_cls_mask) & assigned_gt_cls_mask)  # misclassified mask

                    # Using kitti test class-wise fg threshold instead of thresholds used during train.
                    classwise_fg_thresh = self.min_overlaps[cind]
                    fg_mask = preds_iou_max >= classwise_fg_thresh
                    bg_mask = preds_iou_max <= self.config.ROI_HEAD.TARGET_CONFIG.UNLABELED_CLS_BG_THRESH
                    uc_mask = ~(bg_mask | fg_mask)  # uncertain mask

                    cc_fg_mask = fg_mask & cc_mask
                    classwise_metrics['pred_fgs'][cind] = (cc_fg_mask).sum() / cc_mask.sum()
                    classwise_metrics['pred_ious'][cind] = (preds_iou_max * cc_fg_mask.float()).sum() / cc_fg_mask.sum()
                    cls_score_fg = (valid_pred_scores * cc_fg_mask.float()).sum() / (cc_fg_mask).sum()
                    classwise_metrics['score_fgs'][cind] = cls_score_fg

                    cc_uc_mask = uc_mask & cc_mask
                    classwise_metrics['pred_ucs'][cind] = (cc_uc_mask).sum() / cc_mask.sum()
                    classwise_metrics['pred_ious_ucs'][cind] = (preds_iou_max * cc_uc_mask.float()).sum() / cc_uc_mask.sum()
                    cls_score_uc = (valid_pred_scores * cc_uc_mask.float()).sum() / (cc_uc_mask).sum()
                    classwise_metrics['score_ucs'][cind] = cls_score_uc

                    cls_bg_mask = pred_cls_mask & bg_mask
                    cls_score_bg = (valid_pred_scores * cls_bg_mask.float()).sum() / torch.clamp(bg_mask.float().sum(), min=1.0)
                    classwise_metrics['score_bgs'][cind] = cls_score_bg

                    # Using clamp with min=1 in the denominator makes the final results zero when there's no FG,
                    # while without clamp it is N/A, which makes more sense.
                    if valid_roi_scores is not None:
                        cls_sem_score_fg = (valid_roi_scores * cc_fg_mask.float()).sum() / (cc_fg_mask).sum()
                        classwise_metrics['sem_score_fgs'][cind] = cls_sem_score_fg
                        cls_sem_score_bg = (valid_roi_scores * cls_bg_mask.float()).sum() / cls_bg_mask.float().sum()
                        classwise_metrics['sem_score_bgs'][cind] = cls_sem_score_bg
                        cls_sem_score_uc = (valid_roi_scores * cc_uc_mask.float()).sum() / cc_uc_mask.float().sum()
                        classwise_metrics['sem_score_ucs'][cind] = cls_sem_score_uc

                    if valid_target_scores is not None:
                        cls_target_score_bg = (valid_target_scores * cls_bg_mask.float()).sum() / cls_bg_mask.float().sum()
                        classwise_metrics['target_score_bg'][cind] = cls_target_score_bg
                        cls_target_score_uc = (valid_target_scores * cc_uc_mask.float()).sum() / cc_uc_mask.float().sum()
                        classwise_metrics['target_score_uc'][cind] = cls_target_score_uc

                    if valid_pred_weights is not None:
                        cls_pred_weight_bg = (valid_pred_weights * cls_bg_mask.float()).sum() / cls_bg_mask.float().sum()
                        classwise_metrics['pred_weight_bg'][cind] = cls_pred_weight_bg
                        cls_pred_weight_uc = (valid_pred_weights * cc_uc_mask.float()).sum() / cc_uc_mask.float().sum()
                        classwise_metrics['pred_weight_uc'][cind] = cls_pred_weight_uc
                        cls_pred_weight_fg = (valid_pred_weights * cc_fg_mask.float()).sum() / cc_fg_mask.sum()
                        classwise_metrics['pred_weight_fg'][cind] = cls_pred_weight_fg

                    if valid_pred_iou_wrt_pl is not None:
                        fg_threshs = self.config.ROI_HEAD.TARGET_CONFIG.UNLABELED_CLS_FG_THRESH
                        bg_thresh = self.config.ROI_HEAD.TARGET_CONFIG.UNLABELED_CLS_BG_THRESH
                        classwise_fg_thresh = fg_threshs[cind]
                        fg_mask_wrt_pl = valid_pred_iou_wrt_pl >= classwise_fg_thresh
                        bg_mask_wrt_pl = valid_pred_iou_wrt_pl <= bg_thresh
                        uc_mask_wrt_pl = ~(bg_mask_wrt_pl | fg_mask_wrt_pl)  # uncertain mask

                        cls_fg_mask_wrt_pl = pred_cls_mask & fg_mask_wrt_pl
                        cls_uc_mask_wrt_pl = pred_cls_mask & uc_mask_wrt_pl
                        cls_bg_mask_wrt_pl = pred_cls_mask & bg_mask_wrt_pl
                        # ------ Foreground Mis-classification Metrics ------
                        fn_mask = (cls_bg_mask_wrt_pl | cls_uc_mask_wrt_pl) & cc_fg_mask
                        tp_mask = cls_fg_mask_wrt_pl & cc_fg_mask
                        fp_mask = cls_fg_mask_wrt_pl & (cls_bg_mask | cc_uc_mask)
                        classwise_metrics['pred_fn_rate'][cind] = fn_mask.sum() / cc_fg_mask.sum()
                        classwise_metrics['pred_tp_rate'][cind] = tp_mask.sum() / cc_fg_mask.sum()
                        classwise_metrics['pred_fp_ratio'][cind] = fp_mask.sum() / cls_fg_mask_wrt_pl.sum()
                        classwise_metrics['pred_ious_wrt_pl_fg'][cind] = (valid_pred_iou_wrt_pl * cc_fg_mask.float()).sum() / cc_fg_mask.sum()
                        classwise_metrics['pred_ious_wrt_pl_fn'][cind] = (valid_pred_iou_wrt_pl * fn_mask.float()).sum() / fn_mask.sum()
                        classwise_metrics['pred_ious_wrt_pl_fp'][cind] = (valid_pred_iou_wrt_pl * fp_mask.float()).sum() / fp_mask.sum()
                        classwise_metrics['pred_ious_wrt_pl_tp'][cind] = (valid_pred_iou_wrt_pl * tp_mask.float()).sum() / tp_mask.sum()
                        cls_score_fg_fn = (valid_pred_scores * fn_mask.float()).sum() / fn_mask.sum()
                        cls_score_fg_fp = (valid_pred_scores * fp_mask.float()).sum() / fp_mask.sum()
                        cls_score_fg_tp = (valid_pred_scores * tp_mask.float()).sum() / tp_mask.sum()
                        classwise_metrics['score_fgs_tp'][cind] = cls_score_fg_tp
                        classwise_metrics['score_fgs_fn'][cind] = cls_score_fg_fn
                        classwise_metrics['score_fgs_fp'][cind] = cls_score_fg_fp
                        if valid_target_scores is not None:
                            cls_target_score_fn = (valid_target_scores * fn_mask.float()).sum() / fn_mask.sum()
                            classwise_metrics['target_score_fn'][cind] = cls_target_score_fn
                            cls_target_score_tp = (valid_target_scores * tp_mask.float()).sum() / tp_mask.sum()
                            classwise_metrics['target_score_tp'][cind] = cls_target_score_tp
                            cls_target_score_fp = (valid_target_scores * fp_mask.float()).sum() / fp_mask.sum()
                            classwise_metrics['target_score_fp'][cind] = cls_target_score_fp
                        if valid_pred_weights is not None:
                            cls_pred_weight_fg_mc = (valid_pred_weights * fn_mask.float()).sum() / fn_mask.sum()
                            classwise_metrics['pred_weight_fn'][cind] = cls_pred_weight_fg_mc
                            cls_pred_weight_cc_tp = (valid_pred_weights * tp_mask).sum() / tp_mask.float().sum()
                            classwise_metrics['pred_weight_tp'][cind] = cls_pred_weight_cc_tp
                            cls_pred_weight_cc_fp = (valid_pred_weights * fp_mask).sum() / fp_mask.float().sum()
                            classwise_metrics['pred_weight_fp'][cind] = cls_pred_weight_cc_fp

            for key, val in classwise_metrics.items():
                # Note that unsqueeze is necessary because torchmetric performs the dist cat on dim 0.
                getattr(self, key).append(val.unsqueeze(dim=0))

        # If no prediction is given all states are filled with nan tensors
        if len(preds) == 0:
            for metric_name in self.metrics_name:
                getattr(self, metric_name).append(sample_tensor.new_zeros(num_classes + 1).fill_(float('nan')))

    def compute(self):
        final_results = {}
        if len(self.pred_ious) >= self.reset_state_interval:
            results = {}
            for mname in self.metrics_name:
                mstate = getattr(self, mname)
                if isinstance(mstate, torch.Tensor):
                    mstate = [mstate]
                results[mname] = nanmean(torch.cat(mstate, dim=0), dim=0)  # torch.nanmean is not available in pytorch < 1.8

            for key, val in results.items():
                classwise_results = {}
                for cind, cls in enumerate(self.dataset.class_names + ['cls_agnostic']):
                    if not torch.isnan(val[cind]):
                        classwise_results[cls] = val[cind].item()
                final_results[key] = classwise_results

            # TODO(farzad) Does calling reset in compute make a trouble?
            self.reset()

        return final_results


# TODO(farzad) This class should later be derived from PredQualityMetrics to avoid repeating the code and computation
class KITTIEvalMetrics(Metric):
    full_state_update: bool = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_state_interval = kwargs.get('reset_state_interval', 256)
        self.tag = kwargs.get('tag', None)
        self.dataset = kwargs.get('dataset', None)
        current_classes = self.dataset.class_names
        self.metric = 2  # evaluation only for 3D metric (2)
        overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                                [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
        self.min_overlaps = np.expand_dims(overlap_0_7, axis=0)  # [1, num_metrics, num_cls][1, 3, 6]
        self.class_to_name = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Van', 4: 'Person_sitting', 5: 'Truck'}
        name_to_class = {v: n for n, v in self.class_to_name.items()}
        if not isinstance(current_classes, (list, tuple)):
            current_classes = [current_classes]
        current_classes_int = []
        for curcls in current_classes:
            if isinstance(curcls, str):
                current_classes_int.append(name_to_class[curcls])
            else:
                current_classes_int.append(curcls)
        self.current_classes = current_classes_int
        self.min_overlaps = self.min_overlaps[:, :, self.current_classes]
        self.add_state("detections", default=[])
        self.add_state("groundtruths", default=[])
        self.add_state("overlaps", default=[])

    def update(self, preds: [torch.Tensor], pred_scores: [torch.Tensor], ground_truths: [torch.Tensor],
               rois=None, roi_scores=None, targets=None, target_scores=None, pred_weights=None,
               pseudo_labels=None, pseudo_label_scores=None, iou_wrt_pl=False) -> None:
        if not cfg.MODEL.POST_PROCESSING.ENABLE_KITTI_EVAL:
            return
        assert all([pred.shape[-1] == 8 for pred in preds]) and all([tar.shape[-1] == 8 for tar in ground_truths])
        if roi_scores is not None:
            assert len(pred_scores) == len(roi_scores)
        preds = [pred_box.clone().detach() for pred_box in preds]
        ground_truths = [gt_box.clone().detach() for gt_box in ground_truths]
        pred_scores = [ps_score.clone().detach() for ps_score in pred_scores]
        roi_scores = [score.clone().detach() for score in roi_scores] if roi_scores is not None else None

        for i in range(len(preds)):
            valid_preds_mask = torch.logical_not(torch.all(preds[i] == 0, dim=-1))
            valid_gts_mask = torch.logical_not(torch.all(ground_truths[i] == 0, dim=-1))
            if pred_scores[i].ndim == 1:
                pred_scores[i] = pred_scores[i].unsqueeze(dim=-1)
            if roi_scores is not None and roi_scores[i].ndim == 1:
                roi_scores[i] = roi_scores[i].unsqueeze(dim=-1)

            valid_pred_boxes = preds[i][valid_preds_mask]
            valid_gt_boxes = ground_truths[i][valid_gts_mask]
            valid_pred_scores = pred_scores[i][valid_preds_mask.nonzero().view(-1)]
            # valid_roi_scores = roi_scores[i][valid_preds_mask.nonzero().view(-1)] if roi_scores else None

            # Starting class indices from zero
            valid_pred_boxes[:, -1] -= 1
            valid_gt_boxes[:, -1] -= 1

            # Adding predicted scores as the last column
            valid_pred_boxes = torch.cat([valid_pred_boxes, valid_pred_scores], dim=-1)

            num_gts = valid_gts_mask.sum()
            num_preds = valid_preds_mask.sum()
            overlap = valid_gts_mask.new_zeros((num_preds, num_gts))
            if num_gts > 0 and num_preds > 0:
                overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_pred_boxes[:, 0:7], valid_gt_boxes[:, 0:7])

            self.detections.append(valid_pred_boxes)
            self.groundtruths.append(valid_gt_boxes)
            self.overlaps.append(overlap)

    def compute(self):
        final_results = {}
        if (len(self.detections) >= self.reset_state_interval) and cfg.MODEL.POST_PROCESSING.ENABLE_KITTI_EVAL:
            # eval_class() takes ~45ms for each sample and linearly increasing
            # => ~1.7s for one epoch or 37 samples (if only called once at the end of epoch).
            kitti_eval_metrics = eval_class(self.groundtruths, self.detections, self.current_classes,
                                 self.metric, self.min_overlaps, self.overlaps)
            mAP_3d = get_mAP(kitti_eval_metrics["precision"])
            mAP_3d_R40 = get_mAP_R40(kitti_eval_metrics["precision"])
            kitti_eval_metrics.update({"mAP_3d": mAP_3d, "mAP_3d_R40": mAP_3d_R40})

            # Get calculated TPs, FPs, FNs
            # Early results might not be correct as the 41 values are initialized with zero
            # and only a few predictions are available and thus a few thresholds are non-zero.
            # Therefore, mean over several zero values results in low final value.
            # detailed_stats shape (3, 1, 41, 5) where last dim is
            # {0: 'tp', 1: 'fp', 2: 'fn', 3: 'similarity', 4: 'precision thresholds'}
            total_num_samples = max(len(self.detections), 1)
            detailed_stats = kitti_eval_metrics['detailed_stats']
            raw_metrics_classwise = {}
            for m, metric_name in enumerate(
                    ['tps', 'fps', 'fns', 'sim', 'thresh', 'trans_err', 'orient_err', 'scale_err']):
                if metric_name == 'sim' or metric_name == 'thresh':
                    continue
                class_metrics_all = {}
                class_metrics_batch = {}
                for c, cls_name in enumerate(['Car', 'Pedestrian', 'Cyclist']):
                    metric_value = np.nanmax(detailed_stats[c, 0, :, m])
                    if not np.isnan(metric_value):
                        class_metrics_all[cls_name] = metric_value
                        if metric_name in ['tps', 'fps', 'fns']:
                            class_metrics_batch[cls_name] = metric_value / total_num_samples
                        elif metric_name in ['trans_err', 'orient_err', 'scale_err']:
                            class_metrics_batch[cls_name] = metric_value
                raw_metrics_classwise[metric_name] = class_metrics_all
                if metric_name in ['tps', 'fps', 'fns']:
                    kitti_eval_metrics[metric_name + '_per_sample'] = class_metrics_batch
                elif metric_name in ['trans_err', 'orient_err', 'scale_err']:
                    kitti_eval_metrics[metric_name + '_per_tps'] = class_metrics_batch

            # Get calculated PR and class distribution
            num_lbl_samples = len(self.dataset.kitti_infos)
            num_ulb_samples = total_num_samples
            ulb_lbl_ratio = num_ulb_samples / num_lbl_samples
            pred_labels, pred_scores = [], []
            for sample_dets in self.detections:
                if len(sample_dets) == 0:
                    continue
                pred_labels.append(sample_dets[:, -2])
                pred_scores.append(sample_dets[:, -1])
            pred_labels = torch.cat(pred_labels).to(torch.int64).view(-1)
            pred_scores = torch.cat(pred_scores).view(-1)
            classwise_thresh = pred_scores.new_tensor(self.min_overlaps[0, self.metric]).unsqueeze(0).repeat(
                len(pred_labels), 1).gather(
                dim=-1, index=pred_labels.unsqueeze(-1)).view(-1)
            tp_mask = pred_scores >= classwise_thresh
            pr_cls = {}
            ulb_cls_counter = {}
            # Because self.dataset.class_counter has other classes we only keep
            # current classes in this dict to calculate the sum over all classes.
            lbl_cls_counter = {}
            for cls_id, cls_thresh in zip(self.current_classes, self.min_overlaps[0, self.metric]):
                cls_name = self.class_to_name[cls_id]
                num_lbl_cls = self.dataset.class_counter[cls_name]
                cls_mask = pred_labels == cls_id
                num_ulb_cls = (tp_mask & cls_mask).sum().item()
                ulb_cls_counter[cls_name] = num_ulb_cls
                lbl_cls_counter[cls_name] = num_lbl_cls
                pr_cls[cls_name] = num_ulb_cls / (ulb_lbl_ratio * num_lbl_cls)

            cls_dist = {}
            for c, cls_name in enumerate(['Car', 'Pedestrian', 'Cyclist']):
                cls_dist[cls_name+'_lbl'] = lbl_cls_counter[cls_name] / sum(lbl_cls_counter.values())
                cls_dist[cls_name+'_ulb'] = ulb_cls_counter[cls_name] / sum(ulb_cls_counter.values())
            lbl_dist = torch.tensor(list(lbl_cls_counter.values())) / sum(lbl_cls_counter.values())
            ulb_dist = torch.tensor(list(ulb_cls_counter.values())) / sum(ulb_cls_counter.values())

            kl_div = F.kl_div(ulb_dist.log().unsqueeze(0), lbl_dist.unsqueeze(0), reduction="batchmean").item()
            kitti_eval_metrics['class_distribution'] = cls_dist
            kitti_eval_metrics['kl_div'] = kl_div
            kitti_eval_metrics['PR'] = pr_cls

            # Get calculated Precision
            for m, metric_name in enumerate(['mAP_3d', 'mAP_3d_R40']):
                class_metrics_all = {}
                for c, cls_name in enumerate(['Car', 'Pedestrian', 'Cyclist']):
                    metric_value = kitti_eval_metrics[metric_name][c].item()
                    if not np.isnan(metric_value):
                        class_metrics_all[cls_name] = metric_value
                kitti_eval_metrics[metric_name] = class_metrics_all

            # Get calculated recall
            class_metrics_all = {}
            for c, cls_name in enumerate(['Car', 'Pedestrian', 'Cyclist']):
                metric_value = np.nanmax(kitti_eval_metrics['raw_recall'][c])
                if not np.isnan(metric_value):
                    class_metrics_all[cls_name] = metric_value
            kitti_eval_metrics['max_recall'] = class_metrics_all

            # Draw Precision-Recall curves
            fig, axs = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'wspace': 0.5})
            # plt.tight_layout()
            for c, cls_name in enumerate(['Car', 'Pedestrian', 'Cyclist']):
                thresholds = kitti_eval_metrics['detailed_stats'][c, 0, ::-1, 4]
                prec = kitti_eval_metrics['raw_precision'][c, 0, ::-1]
                rec = kitti_eval_metrics['raw_recall'][c, 0, ::-1]
                valid_mask = ~((rec == 0) | (prec == 0))

                ax_c = axs[c]
                ax_c_twin = ax_c.twinx()
                ax_c.plot(thresholds[valid_mask], prec[valid_mask], 'b-')
                ax_c_twin.plot(thresholds[valid_mask], rec[valid_mask], 'r-')
                ax_c.set_title(cls_name)
                ax_c.set_xlabel('Foreground score')
                ax_c.set_ylabel('Precision', color='b')
                ax_c_twin.set_ylabel('Recall', color='r')

            prec_rec_fig = fig.get_figure()
            kitti_eval_metrics['prec_rec_fig'] = prec_rec_fig

            kitti_eval_metrics.pop('recall')
            kitti_eval_metrics.pop('precision')
            kitti_eval_metrics.pop('raw_recall')
            kitti_eval_metrics.pop('raw_precision')
            kitti_eval_metrics.pop('detailed_stats')

            final_results.update(kitti_eval_metrics)
            # TODO(farzad) Does calling reset in compute make a trouble?
            self.reset()

        for key, val in final_results.items():
            if isinstance(val, list):
                final_results[key] = np.nanmean(val)

        return final_results

# https://github.com/pytorch/pytorch/issues/21987#issuecomment-813859270
def nanmean(v: torch.Tensor, *args, allnan=np.nan, **kwargs) -> torch.Tensor:
    """
    :param v: tensor to take mean
    :param dim: dimension(s) over which to take the mean
    :param allnan: value to use in case all values averaged are NaN.
        Defaults to np.nan, consistent with np.nanmean.
    :return: mean.
    """
    def isnan(v):
        if v.dtype is torch.long:
            return v == torch.tensor(np.nan).long()
        else:
            return torch.isnan(v)
    v = v.clone()
    is_nan = isnan(v)
    v[is_nan] = 0

    if np.isnan(allnan):
        return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
    else:
        sum_nonnan = v.sum(*args, **kwargs)
        n_nonnan = float(~is_nan).sum(*args, **kwargs)
        mean_nonnan = torch.zeros_like(sum_nonnan) + allnan
        any_nonnan = n_nonnan > 1
        mean_nonnan[any_nonnan] = (
                sum_nonnan[any_nonnan] / n_nonnan[any_nonnan])
        return mean_nonnan

def eval_class(gt_annos,
               dt_annos,
               current_classes,
               metric,
               min_overlaps,
               overlaps):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    # TODO(farzad) Assuming class labels in gt and dt start from 0

    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    precision = np.nan * np.zeros([num_class, num_minoverlap, N_SAMPLE_PTS])
    recall = np.nan * np.zeros([num_class, num_minoverlap, N_SAMPLE_PTS])
    detailed_stats = np.nan * np.zeros([num_class, num_minoverlap, N_SAMPLE_PTS, 8])  # TP, FP, FN, Similarity, thresholds
    raw_precision = np.nan * np.zeros_like(precision)
    raw_recall = np.nan * np.zeros_like(recall)

    for m, current_class in enumerate(current_classes):
        rets = _prepare_data(gt_annos, dt_annos, current_class)
        (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
         dontcares, total_dc_num, total_num_valid_gt) = rets
        for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
            thresholdss = []
            for i in range(len(gt_annos)):
                rets = compute_statistics_jit(overlaps[i], gt_datas_list[i], dt_datas_list[i], ignored_gts[i],
                                              ignored_dets[i], dontcares[i], metric, min_overlap=min_overlap,
                                              thresh=0.0, compute_fp=False)
                tp, fp, fn, similarity, thresholds, *_ = rets
                thresholdss += thresholds.tolist()
            thresholdss = np.array(thresholdss)
            thresholds = get_thresholds(thresholdss, total_num_valid_gt)
            thresholds = np.array(thresholds)
            pr = np.zeros([len(thresholds), 7])
            for i in range(len(gt_annos)):
                for t, thresh in enumerate(thresholds):
                    tp, fp, fn, similarity, _, tp_indices, gt_indices = compute_statistics_jit(overlaps[i], gt_datas_list[i],
                                                                       dt_datas_list[i], ignored_gts[i],
                                                                       ignored_dets[i], dontcares[i],
                                                                       metric, min_overlap=min_overlap,
                                                                       thresh=thresh, compute_fp=True)
                    if 0 < tp:
                        assignment_err = cal_tp_metric(dt_datas_list[i][tp_indices], gt_datas_list[i][gt_indices])
                        pr[t, 4] += assignment_err[0]
                        pr[t, 5] += assignment_err[1]
                        pr[t, 6] += assignment_err[2]

                    pr[t, 0] += tp
                    pr[t, 1] += fp
                    pr[t, 2] += fn
                    if similarity != -1:
                        pr[t, 3] += similarity

            for i in range(len(thresholds)):
                detailed_stats[m, k, i, 0] = pr[i, 0]
                detailed_stats[m, k, i, 1] = pr[i, 1]
                detailed_stats[m, k, i, 2] = pr[i, 2]
                detailed_stats[m, k, i, 3] = pr[i, 3]
                detailed_stats[m, k, i, 4] = thresholds[i]
                detailed_stats[m, k, i, 5] = pr[i, 4] / pr[i, 0]
                detailed_stats[m, k, i, 6] = pr[i, 5] / pr[i, 0]
                detailed_stats[m, k, i, 7] = pr[i, 6] / pr[i, 0]

            for i in range(len(thresholds)):
                recall[m, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                precision[m, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])

            raw_recall[m, k] = recall[m, k]
            raw_precision[m, k] = precision[m, k]
            for i in range(len(thresholds)):
                precision[m, k, i] = np.nanmax(
                    precision[m, k, i:], axis=-1)
                recall[m, k, i] = np.nanmax(recall[m, k, i:], axis=-1)

    ret_dict = {
        "recall": recall,
        "precision": precision,
        "detailed_stats": detailed_stats,
        "raw_recall": raw_recall,
        "raw_precision": raw_precision
    }
    return ret_dict


def _prepare_data(gt_annos, dt_annos, current_class):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas_list.append(gt_annos[i])
        dt_datas_list.append(dt_annos[i])
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def clean_data(gt_anno, dt_anno, current_class):
    dc_bboxes, ignored_gt, ignored_dt = [], [], []

    num_gt = len(gt_anno)  # len(gt_anno["name"])
    num_dt = len(dt_anno)  # len(dt_anno["name"])
    num_valid_gt = 0
    # TODO(farzad) cleanup and parallelize
    for i in range(num_gt):
        gt_cls_ind = gt_anno[i][-1]
        if (gt_cls_ind == current_class):
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(-1)

    for i in range(num_dt):
        dt_cls_ind = dt_anno[i][-2]
        if (dt_cls_ind == current_class):
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False):
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    tp_indices = []
    gt_indices = []
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            tp_indices.append(det_idx)
            gt_indices.append(i)
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1

    return tp, fp, fn, similarity, thresholds[:thresh_idx], np.array(tp_indices), np.array(gt_indices)


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


def cal_angle_diff(angle1, angle2):
    # angle is from x to y, anti-clockwise
    angle1 = cor_angle_range(angle1)
    angle2 = cor_angle_range(angle2)

    diff = torch.abs(angle1 - angle2)
    gt_pi_mask = diff > math.pi
    diff[gt_pi_mask] = 2 * math.pi - diff[gt_pi_mask]

    return diff


def cal_tp_metric(tp_boxes, gt_boxes):
    assert tp_boxes.shape[0] == gt_boxes.shape[0]
    # L2 distance xy only
    center_distance = torch.norm(tp_boxes[:, :2] - gt_boxes[:, :2], dim=1)
    trans_err = center_distance.sum().item()

    # Angle difference
    angle_diff = cal_angle_diff(tp_boxes[:, 6], gt_boxes[:, 6])
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

    return trans_err, orient_err, scale_err


def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def get_mAP(prec):
    # sums = 0
    # for i in range(0, prec.shape[-1], 4):
    #     sums = sums + prec[..., i]
    # return sums / 11 * 100
    return np.nanmean(prec[..., ::4], axis=-1) * 100

def get_mAP_R40(prec):
    # sums = 0
    # for i in range(1, prec.shape[-1]):
    #     sums = sums + prec[..., i]
    # return sums / 40 * 100
    return np.nanmean(prec[..., 1:], axis=-1) * 100



def _stats(pred_infos, gt_infos):
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

    pred_infos = pickle.load(open(pred_infos, 'rb'))
    gt_infos = pickle.load(open(gt_infos, 'rb'))
    gt_annos = [info['annos'] for info in gt_infos]
    PR_detail_dict = {}
    ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
        gt_annos, pred_infos, current_classes=['Car', 'Pedestrian', 'Cyclist'], PR_detail_dict=PR_detail_dict
    )

    detailed_stats_3d = PR_detail_dict['3d']['detailed_stats']
    # detailed_stats_3d is a tensor of size [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS, NUM_STATS] where
    # num_class in [0..6], and {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Van', 4: 'Person_sitting', 5: 'Truck'},
    # num_difficulty is 3, and {0: 'easy', 1: 'normal', 2: 'hard'},
    # num_minoverlap is 2, and {0: 'overlap_0_7', 1: 'overlap_0_5'} and overlap_0_7 (for 3D metric),
    # is [0.7, 0.5, 0.5, 0.7, 0.5, 0.7] for 'Car', 'Pedestrian', etc. correspondingly,
    # N_SAMPLE_PTS is 41,
    # NUM_STATS is 5, and {0:'tp', 1:'fp', 2:'fn', 3:'similarity', 4:'precision thresholds'}
    # for example [0, 1, 0, :, 0] means number of TPs of Car class with normal difficulty and overlap@0.7 for all 41 sample points

    # Example of extracting overlap between gts and dets of an example based on specific class and difficulty combination
    example_idx = 1  # second example in our dataset
    class_idx = 0  # class Car
    difficulty_idx = 1  # medium difficulty
    import numpy as np
    overlaps = PR_detail_dict['3d']['overlaps']
    class_difficulty_ignored_gts_mask = PR_detail_dict['3d']['class_difficulty_ignored_gts_mask']
    class_difficulty_ignored_dets_mask = PR_detail_dict['3d']['class_difficulty_ignored_dets_mask']
    valid_gts_inds = np.where(class_difficulty_ignored_gts_mask[class_idx, difficulty_idx, example_idx] == 0)[0]
    valid_dets_inds = np.where(class_difficulty_ignored_dets_mask[class_idx, difficulty_idx, example_idx] == 0)[0]
    valid_inds = np.ix_(valid_dets_inds, valid_gts_inds)
    cls_diff_overlaps = overlaps[example_idx][valid_inds]
    print("cls_diff_overlaps: ", cls_diff_overlaps)
    print("cls_diff_overlaps.shape: ", cls_diff_overlaps.shape)

    # Reproducing fig. 3 of soft-teacher as an example
    from matplotlib import pyplot as plt
    fig, ax1 = plt.subplots()
    precision = PR_detail_dict['3d']['precision']
    recall = PR_detail_dict['3d']['recall']
    thresholds = detailed_stats_3d[0, 1, 0, ::-1, -1]
    prec = precision[0, 1, 0, ::-1]
    rec = recall[0, 1, 0, ::-1]
    ax2 = ax1.twinx()
    valid_mask = ~((rec == 0) | (prec == 0))
    ax1.plot(thresholds[valid_mask], prec[valid_mask], 'b-')
    ax2.plot(thresholds[valid_mask], rec[valid_mask], 'r-')
    ax1.set_xlabel('Foreground score')
    ax1.set_ylabel('Precision', color='b')
    ax2.set_ylabel('Recall', color='r')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    args = parser.parse_args()

    _stats(args.pred_infos, args.gt_infos)


if __name__ == '__main__':
    main()
