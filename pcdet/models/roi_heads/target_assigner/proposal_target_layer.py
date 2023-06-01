import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils


class ProposalTargetLayer(nn.Module):
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
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        code_size = rois.shape[-1]
        
        # Initialize tensors with default values
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_reg_valid_mask = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        batch_cls_labels = -rois.new_ones(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        interval_mask = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, dtype=torch.bool)

        for index in range(batch_size):
            cur_gt_boxes = batch_dict['gt_boxes'][index]
            k = cur_gt_boxes.__len__() - 1
            while k >= 0 and cur_gt_boxes[k].sum() == 0:
                k -= 1
            cur_gt_boxes = cur_gt_boxes[:k + 1]
            cur_gt_boxes = cur_gt_boxes.new_zeros((1, cur_gt_boxes.shape[1])) if len(
                cur_gt_boxes) == 0 else cur_gt_boxes

            if index in batch_dict['unlabeled_inds']:
                # Subsample unlabeled ROIs using Top-K subsampler
                sampled_inds, cur_reg_valid_mask, cur_cls_labels, roi_ious, gt_assignment, cur_interval_mask = \
                    self.subsample_unlabeled_rois(batch_dict, index)
            else:
                # Subsampler labeled ROIs using randomly balanced subsampler
                sampled_inds, cur_reg_valid_mask, cur_cls_labels, roi_ious, gt_assignment, cur_interval_mask = \
                    self.subsample_labeled_rois(batch_dict, index)
            
            # Fill tensors with sampled values
            batch_rois[index] = batch_dict['rois'][index][sampled_inds]
            batch_gt_of_rois[index] = cur_gt_boxes[gt_assignment[sampled_inds]]
            batch_roi_ious[index] = roi_ious
            batch_roi_scores[index] = batch_dict['roi_scores'][index][sampled_inds]
            batch_roi_labels[index] = batch_dict['roi_labels'][index][sampled_inds]
            batch_reg_valid_mask[index] = cur_reg_valid_mask
            batch_cls_labels[index] = cur_cls_labels
            interval_mask[index] = cur_interval_mask
            
        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': batch_reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels,
                        'interval_mask': interval_mask}

        return targets_dict

    def subsample_unlabeled_rois(self, batch_dict, index):
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

        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        bg_rois_per_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_image
        
        # top-k subsampler
        _, sampled_inds = torch.topk(max_overlaps, k=fg_rois_per_image + bg_rois_per_image)
        roi_ious = max_overlaps[sampled_inds]
        sampled_cur_roi_labels = cur_roi_labels[sampled_inds].detach().clone() - 1

        # ---------- regression valid mask ---------- 
        # Fetch thresholds for ROIs based on classwise thresholds (currently threshold is set to class-agnostic 0.55)
        reg_fg_thresh = self.roi_sampler_cfg.UNLABELED_REG_FG_THRESH
        reg_fg_thresh = roi_ious.new_tensor(reg_fg_thresh).reshape(1, -1).repeat(*roi_ious.shape[:2], 1)
        reg_fg_thresh = torch.gather(reg_fg_thresh, dim=-1, index=sampled_cur_roi_labels.unsqueeze(-1)).squeeze(-1)
        cur_reg_valid_mask = (roi_ious > reg_fg_thresh).long()

        # ---------- classification labels ---------- 
        # Fetch thresholds for ROIs based on classwise thresholds
        cls_fg_thresh = self.roi_sampler_cfg.UNLABELED_CLS_FG_THRESH
        fg_thresh = roi_ious.new_tensor(cls_fg_thresh).reshape(1, -1,).repeat(*roi_ious.shape[:2], 1)
        cls_fg_thresh = torch.gather(fg_thresh, dim=-1, index=sampled_cur_roi_labels.unsqueeze(-1)).squeeze(-1)
        cls_bg_thresh = self.roi_sampler_cfg.UNLABELED_CLS_BG_THRESH
        
        fg_mask = roi_ious > cls_fg_thresh
        bg_mask = roi_ious < cls_bg_thresh
        interval_mask = ~(fg_mask | bg_mask)
        ignore_mask = torch.eq(roi_ious, 0).all(dim=-1)

        # initialize classification labels with -1s (invalid)
        cur_cls_labels = -torch.ones_like(sampled_inds, dtype=torch.float)
        # Hard labeling for FGs/BGs, soft labeling for UCs
        cur_cls_labels[ignore_mask] = -1
        cur_cls_labels[fg_mask] = 1.
        cur_cls_labels[bg_mask] = 0.
        cur_cls_labels[interval_mask] = (roi_ious[interval_mask] - cls_bg_thresh) \
                                        / (cls_fg_thresh[interval_mask] - cls_bg_thresh)

        return sampled_inds, cur_reg_valid_mask, cur_cls_labels, roi_ious, gt_assignment, interval_mask

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

        # ---------- regression valid mask ----------
        reg_valid_mask = (roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()

        # ---------- classification labels ----------
        fg_mask = roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH
        bg_mask = roi_ious < self.roi_sampler_cfg.CLS_BG_THRESH
        interval_mask = (fg_mask == 0) & (bg_mask == 0)
        cls_labels = (fg_mask > 0).float()
        cls_labels[interval_mask] = \
            (roi_ious[interval_mask] - self.roi_sampler_cfg.CLS_BG_THRESH) \
                / (self.roi_sampler_cfg.CLS_FG_THRESH - self.roi_sampler_cfg.CLS_BG_THRESH)

        return sampled_inds, reg_valid_mask, cls_labels, roi_ious, gt_assignment, interval_mask

    def subsample_rois(self, max_overlaps, reg_fg_thresh=None, cls_fg_thresh=None):
        if reg_fg_thresh is None:
            reg_fg_thresh = self.roi_sampler_cfg.REG_FG_THRESH
        if cls_fg_thresh is None:
            cls_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        if isinstance(reg_fg_thresh, torch.Tensor):
            fg_thresh = torch.min(reg_fg_thresh, cls_fg_thresh)
        else:
            fg_thresh = min(reg_fg_thresh, cls_fg_thresh)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)  # > 0.55
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)  # < 0.1
        hard_bg_inds = ((max_overlaps < fg_thresh) &
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
