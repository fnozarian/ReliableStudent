import copy
import os

import torch
import torch.nn.functional as F
from pcdet.datasets.augmentor.augmentor_utils import *
from pcdet.ops.iou3d_nms import iou3d_nms_utils

from ...utils import common_utils
from .detector3d_template import Detector3DTemplate

from.pv_rcnn import PVRCNN


def _mean(tensor_list):
    tensor = torch.cat(tensor_list)
    tensor = tensor[~torch.isnan(tensor)]
    mean = tensor.mean() if len(tensor) > 0 else torch.tensor([float('nan')])
    return mean


def _to_dict_of_tensors(list_of_dicts, agg_mode='stack'):
    new_dict = {}
    for k in list_of_dicts[0].keys():
        vals = []
        for i in range(len(list_of_dicts)):
            vals.append(list_of_dicts[i][k])
        agg_vals = torch.cat(vals, dim=0) if agg_mode == 'cat' else torch.stack(vals, dim=0)
        new_dict[k] = agg_vals
    return new_dict


def _to_list_of_dicts(dict_of_tensors, batch_size):
    new_list = []
    for batch_index in range(batch_size):
        inner_dict = {}
        for key in dict_of_tensors.keys():
            assert dict_of_tensors[key].shape[0] == batch_size
            inner_dict[key] = dict_of_tensors[key][batch_index]
        new_list.append(inner_dict)

    return new_list


def _mean_and_var(batch_dict_a, batch_dict_b, unlabeled_inds, keys=()):
    # !!! Note that the function is inplace !!!
    for k in keys:
        batch_dict_mean_k = torch.zeros_like(batch_dict_a[k])
        batch_dict_emas = torch.stack([batch_dict_a[k][unlabeled_inds], batch_dict_b[k][unlabeled_inds]], dim=-1)
        batch_dict_mean_k[unlabeled_inds] = torch.mean(batch_dict_emas, dim=-1)
        batch_dict_a[k + '_mean'] = batch_dict_mean_k
        batch_dict_var_k = torch.zeros_like(batch_dict_a[k])
        batch_dict_var_k[unlabeled_inds] = torch.var(batch_dict_emas, dim=-1)
        batch_dict_a[k + '_var'] = batch_dict_var_k


def _normalize_scores(batch_dict, score_keys=('batch_cls_preds',)):
    # !!! Note that the function is inplace !!!
    assert all([key in ['batch_cls_preds', 'roi_scores'] for key in score_keys])
    for score_key in score_keys:
        if score_key == 'batch_cls_preds':
            if not batch_dict['cls_preds_normalized']:
                batch_dict[score_key] = torch.sigmoid(batch_dict[score_key])
                batch_dict['cls_preds_normalized'] = True
        else:
            batch_dict[score_key] = torch.sigmoid(batch_dict[score_key])

# TODO(farzad) should be tested and debugged
def _weighted_mean(batch_dict_a, batch_dict_b, unlabeled_inds, score_key='batch_cls_preds', keys=()):
    assert score_key in ['batch_cls_preds', 'roi_scores']
    _normalize_scores(batch_dict_a, score_keys=(score_key,))
    _normalize_scores(batch_dict_b, score_keys=(score_key,))
    scores_a = batch_dict_a[score_key][unlabeled_inds]
    scores_b = batch_dict_b[score_key][unlabeled_inds]
    weights = scores_a / (scores_a + scores_b)

    for k in keys:
        batch_dict_mean_k = torch.zeros_like(batch_dict_a[k])
        batch_dict_mean_k[unlabeled_inds] = weights * batch_dict_a[k][unlabeled_inds] + \
                                          (1 - weights) * batch_dict_b[k][unlabeled_inds]
        batch_dict_a[k + '_mean'] = batch_dict_mean_k

# TODO(farzad) should be tested and debugged
def _max_score_replacement(batch_dict_a, batch_dict_b, unlabeled_inds, score_key='batch_cls_preds', keys=()):
    # !!! Note that the function is inplace !!!
    assert score_key in ['batch_cls_preds', 'roi_scores']
    _normalize_scores(batch_dict_a, score_keys=(score_key,))
    _normalize_scores(batch_dict_b, score_keys=(score_key,))
    batch_dict_cat = torch.stack([batch_dict_a[score_key], batch_dict_b[score_key]], dim=-1)
    max_inds = torch.argmax(batch_dict_cat, dim=-1)
    for key in keys:
        batch_dict_a[key][unlabeled_inds] = batch_dict_cat[key][unlabeled_inds, ..., max_inds]


class PVRCNN_SSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # something changes so need deep copy
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        self.pv_rcnn = PVRCNN(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)

        # self.module_list = self.build_networks()
        # self.module_list_ema = self.build_networks()
        self.thresh = model_cfg.THRESH
        self.sem_thresh = model_cfg.SEM_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE

    def forward(self, batch_dict):
        if self.training:
            labeled_mask = batch_dict['labeled_mask'].view(-1)
            labeled_inds = torch.nonzero(labeled_mask).squeeze(1).long()
            unlabeled_inds = torch.nonzero(1-labeled_mask).squeeze(1).long()
            batch_dict['unlabeled_inds'] = unlabeled_inds
            batch_dict_ema = {}
            keys = list(batch_dict.keys())
            for k in keys:
                if k + '_ema' in keys:
                    continue
                if k.endswith('_ema'):
                    batch_dict_ema[k[:-4]] = batch_dict[k]
                else:
                    batch_dict_ema[k] = batch_dict[k]
            
            # If ENABLE_RELIABILITY is True, run WA (Humble Teacher) along with original teacher 
            if self.model_cfg['ROI_HEAD'].get('ENABLE_RELIABILITY', False):
                # Create new dict for weakly aug.(WA) data for teacher - Eg. flip along x axis
                batch_dict_ema_wa = {}
                keys = list(batch_dict.keys())
                for k in keys:
                    if k + '_ema_wa' in keys:
                        continue
                    if k.endswith('_ema_wa'):
                        batch_dict_ema_wa[k[:-7]] = batch_dict[k]
                    else:
                        # TODO(farzad) Warning! Here flip_x values are copied from _ema to _ema_wa which is not correct!
                        batch_dict_ema_wa[k] = batch_dict[k]

                with torch.no_grad():
                    self.pv_rcnn_ema.train()
                    for cur_module in self.pv_rcnn_ema.module_list:
                        # Do not use RPN to produce rois for WA image, instead augment (eg. flip) 
                        # the proposal coord. of P horizontally to obtain P^
                        try:
                            batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                            if cur_module.model_cfg['NAME'] == 'AnchorHeadSingle':
                                # Use proposals generated from original (non-augmented) input
                                # to pool features generated from weakly-augmented input
                                # Note that the proposals should be (weakly-) augmented before pooling!
                                batch_dict_ema_wa['batch_cls_preds'] = batch_dict_ema[
                                    'batch_cls_preds'].clone().detach()
                                batch_dict_ema_wa['batch_box_preds'] = batch_dict_ema[
                                    'batch_box_preds'].clone().detach()
                                batch_dict_ema_wa['cls_preds_normalized'] = batch_dict_ema['cls_preds_normalized']

                                enable = [1] * len(unlabeled_inds)
                                batch_dict_ema_wa['batch_box_preds'][unlabeled_inds] = random_flip_along_x_bbox(
                                    batch_dict_ema_wa['batch_box_preds'][unlabeled_inds],
                                    enables=enable)
                            else:
                                batch_dict_ema_wa = cur_module(batch_dict_ema_wa, disable_gt_roi_when_pseudo_labeling=True)
                        except:
                            # TODO(farzad) we can concat both batch_dict_ema and batch_dict_ema_wa and
                            #  do a forward pass once. Requires more GPU memory, but faster!
                            batch_dict_ema = cur_module(batch_dict_ema)
                            batch_dict_ema_wa = cur_module(batch_dict_ema_wa)
                    # Reverse preds of wa input to match their original (no-aug) preds
                    batch_dict_ema_wa['batch_box_preds'][unlabeled_inds] = random_flip_along_x_bbox(
                        batch_dict_ema_wa['batch_box_preds'][unlabeled_inds], [1] * len(unlabeled_inds))

                    # pseudo-labels used for training rpn head
                    pred_dicts_ens = self.ensemble_post_processing(batch_dict_ema, batch_dict_ema_wa, unlabeled_inds,
                                                                   ensemble_option='mean_pre_nms')
                    if self.model_cfg['ROI_HEAD'].get('ENABLE_RELIABILITY', False):
                        # pseudo-labels used for training roi head
                        # TODO(farzad) BUG! only unlabeled data should have no nms. Currently both (un)labeled have no nms.
                        pred_dicts_ens_no_nms = self.ensemble_post_processing(batch_dict_ema, batch_dict_ema_wa,
                                                                              unlabeled_inds,
                                                                              ensemble_option='mean_no_nms')
            # Else, run the original teacher only
            else:
                with torch.no_grad():
                    for cur_module in self.pv_rcnn_ema.module_list:
                        try:
                            batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                        except:
                            batch_dict_ema = cur_module(batch_dict_ema)
                pred_dicts_ens, recall_dicts_ema = self.pv_rcnn_ema.post_processing(batch_dict_ema, no_recall_dict=True,
                                                                                    override_thresh=0.0,
                                                                                    no_nms=self.no_nms)

            pseudo_boxes, pseudo_scores, pseudo_sem_scores, pseudo_boxes_var, pseudo_scores_var = \
                self._filter_pseudo_labels(pred_dicts_ens, unlabeled_inds)

            ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_inds, ...]

            self._fill_with_pseudo_labels(batch_dict, pseudo_boxes, unlabeled_inds, labeled_inds)

            # apply student's augs on teacher's pseudo-labels only (not points)
            batch_dict['gt_boxes'][unlabeled_inds, ...] = random_flip_along_x_bbox(
                batch_dict['gt_boxes'][unlabeled_inds, ...], batch_dict['flip_x'][unlabeled_inds, ...]
            )

            batch_dict['gt_boxes'][unlabeled_inds, ...] = random_flip_along_y_bbox(
                batch_dict['gt_boxes'][unlabeled_inds, ...], batch_dict['flip_y'][unlabeled_inds, ...]
            )

            batch_dict['gt_boxes'][unlabeled_inds, ...] = global_rotation_bbox(
                batch_dict['gt_boxes'][unlabeled_inds, ...], batch_dict['rot_angle'][unlabeled_inds, ...]
            )

            batch_dict['gt_boxes'][unlabeled_inds, ...] = global_scaling_bbox(
                batch_dict['gt_boxes'][unlabeled_inds, ...], batch_dict['scale'][unlabeled_inds, ...]
            )

            pseudo_ious = []
            pseudo_accs = []
            pseudo_fgs = []
            sem_score_fgs = []
            sem_score_bgs = []
            for i, ind in enumerate(unlabeled_inds):
                # statistics
                anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
                    batch_dict['gt_boxes'][ind, ...][:, 0:7],
                    ori_unlabeled_boxes[i, :, 0:7])
                cls_pseudo = batch_dict['gt_boxes'][ind, ...][:, 7]
                nonzero_inds = torch.nonzero(cls_pseudo).squeeze(1).long()
                cls_pseudo = cls_pseudo[nonzero_inds]
                if len(nonzero_inds) > 0:
                    iou_max, asgn = anchor_by_gt_overlap[nonzero_inds, :].max(dim=1)
                    pseudo_ious.append(iou_max.mean().unsqueeze(dim=0))
                    acc = (ori_unlabeled_boxes[i][:, 7].gather(dim=0, index=asgn) == cls_pseudo).float().mean()
                    pseudo_accs.append(acc.unsqueeze(0))
                    fg_thresh = self.model_cfg['ROI_HEAD']['TARGET_CONFIG']['CLS_FG_THRESH']
                    bg_thresh = self.model_cfg['ROI_HEAD']['TARGET_CONFIG']['CLS_BG_THRESH']  # bg_thresh includes both easy and hard bgs
                    fg = (iou_max > fg_thresh).float().sum(dim=0, keepdim=True) / len(nonzero_inds)

                    sem_score_fg = (pseudo_sem_scores[i][nonzero_inds] * (iou_max > fg_thresh).float()).sum(dim=0, keepdim=True) \
                                    / torch.clamp((iou_max > fg_thresh).float().sum(dim=0, keepdim=True), min=1.0)
                    sem_score_bg = (pseudo_sem_scores[i][nonzero_inds] * (iou_max < bg_thresh).float()).sum(dim=0, keepdim=True) \
                                    / torch.clamp((iou_max < bg_thresh).float().sum(dim=0, keepdim=True), min=1.0)
                    pseudo_fgs.append(fg)
                    sem_score_fgs.append(sem_score_fg)
                    sem_score_bgs.append(sem_score_bg)

                    # only for 100% label
                    if self.supervise_mode >= 1:
                        filter = iou_max > 0.3
                        asgn = asgn[filter]
                        batch_dict['gt_boxes'][ind, ...][:] = torch.zeros_like(batch_dict['gt_boxes'][ind, ...][:])
                        batch_dict['gt_boxes'][ind, ...][:len(asgn)] = ori_unlabeled_boxes[i, :].gather(dim=0, index=asgn.unsqueeze(-1).repeat(1, 8))

                        if self.supervise_mode == 2:
                            batch_dict['gt_boxes'][ind, ...][:len(asgn), 0:3] += 0.1 * torch.randn((len(asgn), 3), device=iou_max.device) * \
                                                                                    batch_dict['gt_boxes'][ind, ...][
                                                                                    :len(asgn), 3:6]
                            batch_dict['gt_boxes'][ind, ...][:len(asgn), 3:6] += 0.1 * torch.randn((len(asgn), 3), device=iou_max.device) * \
                                                                                    batch_dict['gt_boxes'][ind, ...][
                                                                                    :len(asgn), 3:6]
                else:
                    nan = torch.tensor([float('nan')], device=unlabeled_inds.device)
                    sem_score_fgs.append(nan)
                    sem_score_bgs.append(nan)
                    pseudo_ious.append(nan)
                    pseudo_accs.append(nan)
                    pseudo_fgs.append(nan)

            for cur_module in self.pv_rcnn.module_list:
                if cur_module.model_cfg['NAME'] == 'PVRCNNHead' and self.model_cfg['ROI_HEAD'].get('ENABLE_RELIABILITY', False):
                    # Pass teacher's proposal to the student.
                    # To let proposal_layer continues for labeled data we pass rois with _ema postfix
                    batch_dict['rois_ema'] = batch_dict_ema['rois'].detach().clone()
                    batch_dict['roi_scores_ema'] = batch_dict_ema['roi_scores'].detach().clone()
                    batch_dict['roi_labels_ema'] = batch_dict_ema['roi_labels'].detach().clone()
                    batch_dict = self.apply_augmentation(batch_dict, batch_dict, unlabeled_inds, key='rois_ema')
                    boxes, labels, scores, sem_scores, boxes_var, scores_var = self._unpack_predictions(pred_dicts_ens_no_nms, unlabeled_inds)
                    pseudo_boxes = [torch.cat([box, label.unsqueeze(-1)], dim=-1) for box, label in zip(boxes, labels)]
                    batch_dict['rcnn_cls_labels'] = torch.zeros_like(batch_dict['roi_scores_ema'])
                    batch_dict['rcnn_cls_labels_var'] = torch.zeros_like(batch_dict['roi_scores_ema'])
                    batch_dict['gt_of_rois_var'] = torch.zeros_like(batch_dict['rois_ema'])
                    for i, ui in enumerate(unlabeled_inds):
                        batch_dict['rcnn_cls_labels'][ui] = scores[i]
                        batch_dict['rcnn_cls_labels_var'][ui] = scores_var[i]
                        batch_dict['gt_of_rois_var'] = boxes_var[i]

                    self._fill_with_pseudo_labels(batch_dict, pseudo_boxes, unlabeled_inds, labeled_inds)

                batch_dict = cur_module(batch_dict)

            disp_dict = {}
            loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

            if not self.unlabeled_supervise_cls:
                loss_rpn_cls = loss_rpn_cls[labeled_inds, ...].mean()
            else:
                loss_rpn_cls = loss_rpn_cls[labeled_inds, ...].mean() + loss_rpn_cls[unlabeled_inds, ...].mean() * self.unlabeled_weight

            loss_rpn_box = loss_rpn_box[labeled_inds, ...].mean() + loss_rpn_box[unlabeled_inds, ...].mean() * self.unlabeled_weight
            loss_point = loss_point[labeled_inds, ...].mean()
            if self.model_cfg.get('UNLABELED_SUPERVISE_OBJ', False):
                loss_rcnn_cls = loss_rcnn_cls[labeled_inds, ...].mean() + loss_rcnn_cls[unlabeled_inds, ...].mean() * self.unlabeled_weight
            else:
                loss_rcnn_cls = loss_rcnn_cls[labeled_inds, ...].mean()
            if not self.unlabeled_supervise_refine:
                loss_rcnn_box = loss_rcnn_box[labeled_inds, ...].mean()
            else:
                loss_rcnn_box = loss_rcnn_box[labeled_inds, ...].mean() + loss_rcnn_box[unlabeled_inds, ...].mean() * self.unlabeled_weight

            loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box
            tb_dict_ = {}
            for key in tb_dict.keys():
                if 'loss' in key:
                    tb_dict_[key+"_labeled"] = tb_dict[key][labeled_inds, ...].mean()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_inds, ...].mean()
                elif 'acc' in key:
                    tb_dict_[key+"_labeled"] = tb_dict[key][labeled_inds, ...].mean()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_inds, ...].mean()
                elif 'point_pos_num' in key:
                    tb_dict_[key + "_labeled"] = tb_dict[key][labeled_inds, ...].mean()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_inds, ...].mean()
                else:
                    tb_dict_[key] = tb_dict[key]

            tb_dict_['pseudo_ious'] = _mean(pseudo_ious)
            tb_dict_['pseudo_accs'] = _mean(pseudo_accs)
            tb_dict_['sem_score_fg'] = _mean(sem_score_fgs)
            tb_dict_['sem_score_bg'] = _mean(sem_score_bgs)

            tb_dict_['max_box_num'] = max(len(box) for box in ori_unlabeled_boxes)
            tb_dict_['max_pseudo_box_num'] = max(len(box) for box in batch_dict['gt_boxes'])

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict_, disp_dict

        else:
            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)

            return pred_dicts, recall_dicts, {}

    def ensemble_post_processing(self, batch_dict_a, batch_dict_b, unlabeled_inds, ensemble_option=None):
        # TODO(farzad) what about roi_labels and roi_scores in following options?
        ens_pred_dicts = None
        if ensemble_option == 'joint_nms':
            cat_keys = ['batch_cls_preds', 'batch_box_preds', 'roi_labels', 'roi_scores']
            for key in cat_keys:
                batch_dict_a[key][unlabeled_inds] = torch.cat(
                    (batch_dict_a[key][unlabeled_inds], batch_dict_b[key][unlabeled_inds]), dim=1)

        elif ensemble_option == 'mean_pre_nms':
            _mean_and_var(batch_dict_a, batch_dict_b, unlabeled_inds,
                          keys=('batch_cls_preds', 'batch_box_preds'))
            # backup original values and replace them with mean values
            for key in ['batch_box_preds', 'batch_cls_preds']:
                batch_dict_a[key + '_src'] = batch_dict_a[key].clone().detach()
                batch_dict_a[key][unlabeled_inds] = batch_dict_a[key + '_mean'][unlabeled_inds]

            ens_pred_dicts, _ = self.pv_rcnn_ema.post_processing(batch_dict_a, no_recall_dict=True)

            # replace means with original values and remove means/vars
            for key in ['batch_box_preds', 'batch_cls_preds']:
                batch_dict_a[key] = batch_dict_a[key + '_src'].clone().detach()
                batch_dict_a.pop(key + '_src')
                batch_dict_a.pop(key + '_mean')
                batch_dict_a.pop(key + '_var')

        elif ensemble_option == 'mean_no_nms':
            # no_nms has been set to True to avoid the filtering and keep the o/p consistent with that of student

            pred_dicts_a, _ = self.pv_rcnn_ema.post_processing(batch_dict_a, no_recall_dict=True, no_nms=True)
            pred_dicts_b, _ = self.pv_rcnn_ema.post_processing(batch_dict_b, no_recall_dict=True, no_nms=True)
            pred_dicts_a = _to_dict_of_tensors(pred_dicts_a)
            pred_dicts_b = _to_dict_of_tensors(pred_dicts_b)
            _mean_and_var(pred_dicts_a, pred_dicts_b, unlabeled_inds, keys=('pred_scores', 'pred_boxes'))
            # replace original values with mean values
            for key in ['pred_scores', 'pred_boxes']:
                pred_dicts_a[key][unlabeled_inds] = pred_dicts_a[key + '_mean'][unlabeled_inds]
                pred_dicts_a.pop(key + '_mean')
            ens_pred_dicts = _to_list_of_dicts(pred_dicts_a, batch_size=batch_dict_a['batch_size'])

        elif ensemble_option == 'weighted_mean':
            _weighted_mean(batch_dict_a, batch_dict_b, unlabeled_inds, keys=('batch_cls_preds', 'batch_box_preds'))

        elif ensemble_option == 'max_only':
            _max_score_replacement(batch_dict_a, batch_dict_b, unlabeled_inds,
                                   keys=('batch_cls_preds', 'batch_box_preds'))

        elif ensemble_option is None:
            ens_pred_dicts, _ = self.pv_rcnn_ema.post_processing(batch_dict_a, no_recall_dict=True)


        return ens_pred_dicts

    # TODO(farzad) refactor and remove this!
    def _unpack_predictions(self, pred_dicts, unlabeled_inds):
        pseudo_boxes = []
        pseudo_scores = []
        pseudo_sem_scores = []
        pseudo_labels = []
        pseudo_boxes_var = []
        pseudo_scores_var = []
        for ind in unlabeled_inds:
            pseudo_score = pred_dicts[ind]['pred_scores']
            pseudo_box = pred_dicts[ind]['pred_boxes']
            pseudo_label = pred_dicts[ind]['pred_labels']
            pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']
            pseudo_box_var = pred_dicts[ind]['pred_boxes_var']
            pseudo_score_var = pred_dicts[ind]['pred_scores_var']
            if len(pseudo_label) == 0:
                pseudo_boxes.append(pseudo_label.new_zeros((0, 7)).float())
                pseudo_boxes_var.append(pseudo_label.new_zeros((0, 7)).float())
                pseudo_sem_scores.append(pseudo_label.new_zeros((1,)).float())
                pseudo_scores.append(pseudo_label.new_zeros((1,)).float())
                pseudo_scores_var.append(pseudo_label.new_zeros((1,)).float())
                pseudo_labels.append(pseudo_label.new_zeros((1,)).float())
                continue

            pseudo_boxes.append(pseudo_box)
            pseudo_boxes_var.append(pseudo_box_var)
            pseudo_sem_scores.append(pseudo_sem_score)
            pseudo_scores.append(pseudo_score)
            pseudo_scores_var.append(pseudo_score_var)
            pseudo_labels.append(pseudo_label)

        return pseudo_boxes, pseudo_labels, pseudo_scores, pseudo_sem_scores, pseudo_boxes_var, pseudo_scores_var

    # TODO(farzad) refactor and remove this!
    def _filter_pseudo_labels(self, pred_dicts, unlabeled_inds):
        pseudo_boxes = []
        pseudo_scores = []
        pseudo_sem_scores = []
        pseudo_scores_var = []
        pseudo_boxes_var = []
        for pseudo_box, pseudo_label, pseudo_score, pseudo_sem_score, pseudo_box_var, pseudo_score_var in zip(
                *self._unpack_predictions(pred_dicts, unlabeled_inds)):

            conf_thresh = torch.tensor(self.thresh, device=pseudo_label.device).unsqueeze(
                0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label - 1).unsqueeze(-1))

            valid_inds = pseudo_score > conf_thresh.squeeze()

            valid_inds = valid_inds * (pseudo_sem_score > self.sem_thresh[0])

            pseudo_sem_score = pseudo_sem_score[valid_inds]
            pseudo_box = pseudo_box[valid_inds]
            pseudo_label = pseudo_label[valid_inds]
            pseudo_score = pseudo_score[valid_inds]
            pseudo_box_var = pseudo_box_var[valid_inds]
            pseudo_score_var = pseudo_score_var[valid_inds]
            # TODO : Two stage filtering instead of applying NMS
            # Stage1 based on size of bbox, Stage2 is objectness thresholding
            # Note : Two stages happen sequentially, and not independently.
            # vol_boxes = ((pseudo_box[:, 3] * pseudo_box[:, 4] * pseudo_box[:, 5])/torch.abs(pseudo_box[:,2][0])).view(-1)
            # vol_boxes, _ = torch.sort(vol_boxes, descending=True)
            # # Set volume threshold to 10% of the maximum volume of the boxes
            # keep_ind = int(self.model_cfg.PSEUDO_TWO_STAGE_FILTER.MAX_VOL_PROP * len(vol_boxes))
            # keep_vol = vol_boxes[keep_ind]
            # valid_inds = vol_boxes > keep_vol # Stage 1
            # pseudo_sem_score = pseudo_sem_score[valid_inds]
            # pseudo_box = pseudo_box[valid_inds]
            # pseudo_label = pseudo_label[valid_inds]
            # pseudo_score = pseudo_score[valid_inds]

            # valid_inds = pseudo_score > self.model_cfg.PSEUDO_TWO_STAGE_FILTER.THRESH # Stage 2
            # pseudo_sem_score = pseudo_sem_score[valid_inds]
            # pseudo_box = pseudo_box[valid_inds]
            # pseudo_label = pseudo_label[valid_inds]
            # pseudo_score = pseudo_score[valid_inds]

            pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
            pseudo_sem_scores.append(pseudo_sem_score)
            pseudo_scores.append(pseudo_score)
            pseudo_scores_var.append(pseudo_score_var)
            pseudo_boxes_var.append(pseudo_box_var)

        return pseudo_boxes, pseudo_scores, pseudo_sem_scores, pseudo_boxes_var, pseudo_scores_var

    def _fill_with_pseudo_labels(self, batch_dict, pseudo_boxes, unlabeled_inds, labeled_inds):
        max_box_num = batch_dict['gt_boxes'].shape[1]

        max_pseudo_box_num = max([len(ps_box) for ps_box in pseudo_boxes])

        if max_box_num >= max_pseudo_box_num:
            for i, pseudo_box in enumerate(pseudo_boxes):
                diff = max_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                batch_dict['gt_boxes'][unlabeled_inds[i]] = pseudo_box
        else:
            ori_boxes = batch_dict['gt_boxes']
            new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                    device=ori_boxes.device)
            for i, inds in enumerate(labeled_inds):
                diff = max_pseudo_box_num - ori_boxes[inds].shape[0]
                new_box = torch.cat([ori_boxes[inds], torch.zeros((diff, 8), device=ori_boxes[inds].device)], dim=0)
                new_boxes[inds] = new_box
            for i, pseudo_box in enumerate(pseudo_boxes):

                diff = max_pseudo_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                new_boxes[unlabeled_inds[i]] = pseudo_box
            batch_dict['gt_boxes'] = new_boxes

    def apply_augmentation(self, batch_dict, batch_dict_org, unlabeled_inds, key = 'rois'):
        batch_dict[key][unlabeled_inds] = random_flip_along_x_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_x'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = random_flip_along_y_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_y'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = global_rotation_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['rot_angle'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = global_scaling_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['scale'][unlabeled_inds])
        return batch_dict
    
    def reverse_augmentation(self, batch_dict, batch_dict_org, unlabeled_inds, key = 'rois'):
        batch_dict[key][unlabeled_inds] = global_scaling_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['scale'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = global_rotation_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['rot_angle'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = random_flip_along_y_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_y'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = random_flip_along_x_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_x'][unlabeled_inds])
        return batch_dict

    def get_supervised_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def update_global_step(self):
        self.global_step += 1
        alpha = 0.999
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.pv_rcnn_ema.parameters(), self.pv_rcnn.parameters()):
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'pv_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            new_key = 'pv_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
