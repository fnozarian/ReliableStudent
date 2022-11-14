import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils


class ProposalTargetLayerConsistency(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """

        targets_dict = self.sample_rois_for_rcnn(batch_dict=batch_dict)

        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        # Followings will come from teacher for unlabeled samples
        # gt_scores_var = batch_dict['pred_scores_ema_var']
        # gt_boxes_var = batch_dict['pred_boxes_ema_var']
        # gt_pred_iou = batch_dict['pred_ious_ema']

        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_gt_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_reg_valid_mask = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        batch_cls_labels = -rois.new_ones(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)

        for index in range(batch_size):
            cur_gt_boxes = batch_dict['gt_boxes']
            k = cur_gt_boxes.__len__() - 1
            while k >= 0 and cur_gt_boxes[k].sum() == 0:
                k -= 1
            cur_gt_boxes = cur_gt_boxes[:k + 1]
            cur_gt_boxes = cur_gt_boxes.new_zeros((1, cur_gt_boxes.shape[1])) if len(cur_gt_boxes) == 0 else cur_gt_boxes
            if index in batch_dict['unlabeled_inds']:
                subsample_unlabeled_rois = getattr(self, self.roi_sampler_cfg.UNLABELED_SAMPLER_TYPE)
                sampled_inds, cur_reg_valid_mask, cur_cls_labels = subsample_unlabeled_rois(batch_dict, index)

                cur_roi = batch_dict['rois_ema'][index][sampled_inds]
                cur_roi_scores = batch_dict['roi_scores_ema'][index][sampled_inds]
                cur_roi_labels = batch_dict['roi_labels_ema'][index][sampled_inds]
                batch_gt_scores[index] = batch_dict['pred_scores_ema'][index][sampled_inds]
                batch_gt_of_rois[index] = cur_gt_boxes[index][sampled_inds]
            else:
                sampled_inds, cur_reg_valid_mask, cur_cls_labels, roi_ious, gt_assignment = self.subsample_labeled_rois(batch_dict, index)

                cur_roi = batch_dict['rois'][index][sampled_inds]
                cur_roi_scores = batch_dict['roi_scores'][index][sampled_inds]
                cur_roi_labels = batch_dict['roi_labels'][index][sampled_inds]
                batch_roi_ious[index] = roi_ious
                batch_gt_of_rois[index] = cur_gt_boxes[index][gt_assignment[sampled_inds]]

            batch_rois[index] = cur_roi
            batch_roi_labels[index] = cur_roi_labels
            batch_roi_scores[index] = cur_roi_scores

            batch_reg_valid_mask[index] = cur_reg_valid_mask
            batch_cls_labels[index] = cur_cls_labels

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'roi_scores': batch_roi_scores,
                        'roi_labels': batch_roi_labels, 'reg_valid_mask': batch_reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels}

        return targets_dict

    # Localization-based samplers ======================================================================================
    # Should be focused more since the localization loss is x3 significant than the cls loss!

    def bbox_uncertainty_sampler(self, **kwargs):
        raise NotImplementedError

    def pred_ious_sampler(self, **kwargs):
        raise NotImplementedError

    # Confidence-based samplers ========================================================================================
    def classwise_hybrid_thresholds_sampler(self, batch_dict, index):
        # (mis?) using pseudo-label objectness scores as a proxy for iou!

        gt_scores = batch_dict['pred_scores_ema'][index]
        gt_boxes = batch_dict['gt_boxes'][index]
        # TODO(farzad) roi_scores_ema is no longer normalized. Before using the following, please consider normalization
        thresh_inds = (batch_dict['roi_scores_ema'][index] > 0.2).nonzero().view(-1)
        gt_scores = gt_scores[thresh_inds]
        sampled_inds = self.subsample_rois(max_overlaps=gt_scores)
        selected_inds = thresh_inds[sampled_inds]
        sampled_gt_scores = gt_scores[selected_inds]
        sampled_gt_boxes = gt_boxes[selected_inds]
        reg_valid_mask = (sampled_gt_scores > self.roi_sampler_cfg.UNLABELED_REG_FG_THRESH).long()

        iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
        iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
        fg_mask = sampled_gt_scores > iou_fg_thresh
        bg_mask = sampled_gt_scores < iou_bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)
        cls_labels = (fg_mask > 0).float()
        cls_labels[interval_mask] = sampled_gt_scores[interval_mask]
        # Ignoring all-zero pseudo-labels produced due to filtering
        ignore_mask = torch.eq(sampled_gt_boxes, 0).all(dim=-1)
        cls_labels[ignore_mask] = -1

        return selected_inds, reg_valid_mask, cls_labels

    def classwise_adapative_thresholds_sampler(self, **kwargs):
        raise NotImplementedError

    def classwise_top_k_sampler(self, **kwargs):
        roi_labels = kwargs.get('roi_labels')
        gt_scores = kwargs.get('gt_scores')
        classwise_topk_inds = {}
        for k in range(3):  # TODO(Farzad) fixed num class
            roi_mask = (roi_labels == k)
            if roi_mask.sum() > 0:
                cur_gt_scores = gt_scores[roi_mask]
                cur_inds = roi_mask.nonzero().view(-1)
                _, top_k_inds = torch.topk(cur_gt_scores, k=min(100, len(cur_inds)))  # TODO(Farzad) fixed k
                classwise_topk_inds[k] = cur_inds[top_k_inds]
        raise NotImplementedError

    def roi_scores_sampler(self, **kwargs):
        roi_scores = kwargs.get("roi_scores")
        reg_valid_mask = torch.ge(roi_scores, 0.7).long()
        raise NotImplementedError

    def gt_scores_sampler(self, batch_dict, index):
        # (mis?) using pseudo-label objectness scores as a proxy for iou!

        gt_scores = batch_dict['pred_scores_ema'][index]
        gt_boxes = batch_dict['gt_boxes'][index]
        sampled_inds = self.subsample_rois(max_overlaps=gt_scores)
        sampled_gt_scores = gt_scores[sampled_inds]
        sampled_gt_boxes = gt_boxes[sampled_inds]
        reg_valid_mask = (sampled_gt_scores > self.roi_sampler_cfg.UNLABELED_REG_FG_THRESH).long()

        iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
        iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
        fg_mask = sampled_gt_scores > iou_fg_thresh
        bg_mask = sampled_gt_scores < iou_bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)
        cls_labels = (fg_mask > 0).float()
        cls_labels[interval_mask] = sampled_gt_scores[interval_mask]
        # Ignoring all-zero pseudo-labels produced due to filtering
        ignore_mask = torch.eq(sampled_gt_boxes, 0).all(dim=-1)
        cls_labels[ignore_mask] = -1

        return sampled_inds, reg_valid_mask, cls_labels

    def subsample_labeled_rois(self, batch_dict, index):
        cur_roi = batch_dict['rois'][index]
        cur_gt_boxes = batch_dict['gt_boxes'][index]
        cur_roi_labels = batch_dict['roi_labels'][index]

        if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
            max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                rois=cur_roi, roi_labels=cur_roi_labels,
                gt_boxes=cur_gt_boxes[:, 0:7], gt_labels=cur_gt_boxes[:, -1].long()
            )
        else:
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt_boxes[:, 0:7])  # (M, N)
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

        sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)
        roi_ious = max_overlaps[sampled_inds]

        # regression valid mask
        reg_valid_mask = (roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()

        # classification label
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            cls_labels = (roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = roi_ious > iou_fg_thresh
            bg_mask = roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)
            cls_labels = (fg_mask > 0).float()
            cls_labels[interval_mask] = \
                (roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)

        return sampled_inds, reg_valid_mask, cls_labels, roi_ious, gt_assignment

    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)  # > 0.55
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)  # < 0.1
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                        (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0] # yield empty tensor

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment

