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
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        # batch_gt_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_reg_valid_mask = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        batch_cls_labels = -rois.new_ones(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        interval_mask = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, dtype=torch.bool)

        for index in range(batch_size):
            # TODO(farzad) WARNING!!! The index for cur_gt_boxes was missing and caused an error. FIX this in other branches.
            cur_gt_boxes = batch_dict['gt_boxes'][index]
            k = cur_gt_boxes.__len__() - 1
            while k >= 0 and cur_gt_boxes[k].sum() == 0:
                k -= 1
            cur_gt_boxes = cur_gt_boxes[:k + 1]
            cur_gt_boxes = cur_gt_boxes.new_zeros((1, cur_gt_boxes.shape[1])) if len(
                cur_gt_boxes) == 0 else cur_gt_boxes

            if index in batch_dict['unlabeled_inds']:
                subsample_unlabeled_rois = getattr(self, self.roi_sampler_cfg.UNLABELED_SAMPLER_TYPE, None)
                if self.roi_sampler_cfg.UNLABELED_SAMPLER_TYPE is None:
                    sampled_inds, cur_reg_valid_mask, cur_cls_labels, roi_ious, gt_assignment, cur_interval_mask = self.subsample_labeled_rois(batch_dict, index)
                else:
                    sampled_inds, cur_reg_valid_mask, cur_cls_labels, roi_ious, gt_assignment, cur_interval_mask = subsample_unlabeled_rois(batch_dict, index)
                cur_roi = batch_dict['rois'][index][sampled_inds]
                cur_roi_scores = batch_dict['roi_scores'][index][sampled_inds]
                cur_roi_labels = batch_dict['roi_labels'][index][sampled_inds]
                batch_roi_ious[index] = roi_ious
                # batch_gt_scores[index] = batch_dict['pred_scores_ema'][index][sampled_inds]
                batch_gt_of_rois[index] = cur_gt_boxes[gt_assignment[sampled_inds]]
            else:
                sampled_inds, cur_reg_valid_mask, cur_cls_labels, roi_ious, gt_assignment, cur_interval_mask = self.subsample_labeled_rois(batch_dict, index)
                cur_roi = batch_dict['rois'][index][sampled_inds]
                cur_roi_scores = batch_dict['roi_scores'][index][sampled_inds]
                cur_roi_labels = batch_dict['roi_labels'][index][sampled_inds]
                batch_roi_ious[index] = roi_ious
                batch_gt_of_rois[index] = cur_gt_boxes[gt_assignment[sampled_inds]]

            batch_rois[index] = cur_roi
            batch_roi_labels[index] = cur_roi_labels
            batch_roi_scores[index] = cur_roi_scores
            interval_mask[index] = cur_interval_mask
            batch_reg_valid_mask[index] = cur_reg_valid_mask
            batch_cls_labels[index] = cur_cls_labels

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': batch_reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels,
                        'interval_mask': interval_mask}

        return targets_dict

    def subsample_unlabeled_rois_default(self, batch_dict, index):
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

        # TODO(farzad) Define a better sampler! This might limit the full flexibility of pre_loss_filtering!
        # sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        bg_rois_per_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_image
        if self.roi_sampler_cfg.get('UNLABELED_SAMPLE_EASY_BG', False):
            hard_bg_rois_per_image = (int(bg_rois_per_image * self.roi_sampler_cfg.HARD_BG_RATIO))
        else:
            hard_bg_rois_per_image = bg_rois_per_image

        _, sampled_inds = torch.topk(max_overlaps, k=fg_rois_per_image + hard_bg_rois_per_image)

        if self.roi_sampler_cfg.get('UNLABELED_SAMPLE_EASY_BG', False):
            easy_bg_rois_per_image = bg_rois_per_image - hard_bg_rois_per_image
            _, easy_bg_inds = torch.topk(max_overlaps, k=easy_bg_rois_per_image, largest=False)
            sampled_inds = torch.cat([sampled_inds, easy_bg_inds])

        roi_ious = max_overlaps[sampled_inds]

        # interval_mask, reg_valid_mask and cls_labels are defined in pre_loss_filtering based on advanced thresholding.
        cur_reg_valid_mask = torch.zeros_like(sampled_inds, dtype=torch.int)
        cur_cls_labels = -torch.ones_like(sampled_inds, dtype=torch.float)
        interval_mask = torch.zeros_like(sampled_inds, dtype=torch.bool)

        return sampled_inds, cur_reg_valid_mask, cur_cls_labels, roi_ious, gt_assignment, interval_mask

    # TODO(farzad) Our previous method for unlabeled samples. Test it for the new implementation.
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
        iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
        # NOTE (shashank): Use classwise local thresholds used in unlabeled ROIs for labeled ROIs  
        if self.roi_sampler_cfg.USE_ULB_CLS_FG_THRESH_FOR_LB :
            iou_fg_thresh = self.roi_sampler_cfg.UNLABELED_CLS_FG_THRESH
            iou_fg_thresh = roi_ious.new_tensor(iou_fg_thresh).unsqueeze(0).repeat(len(roi_ious), 1)
            iou_fg_thresh = torch.gather(iou_fg_thresh, dim=-1, index=(cur_roi_labels[sampled_inds]-1).unsqueeze(-1)).squeeze(-1)
        else:
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH

        fg_mask = roi_ious > iou_fg_thresh
        bg_mask = roi_ious < iou_bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)
        cls_labels = (fg_mask > 0).float()
        iou_fg_thresh = iou_fg_thresh[interval_mask] if self.roi_sampler_cfg.USE_ULB_CLS_FG_THRESH_FOR_LB else iou_fg_thresh
        cls_labels[interval_mask] = \
            (roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)

        return sampled_inds, reg_valid_mask, cls_labels, roi_ious, gt_assignment, interval_mask

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
