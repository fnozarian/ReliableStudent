# arguments example:
# --pred_infos
# <OpenPCDet_HOME>/output/cfgs/kitti_models/pv_rcnn_ssl/enabled_st_all_bs8_dist4_split_1_2_trial3_169035d/eval/eval_with_train/epoch_60/val/result.pkl
# --gt_infos
# <OpenPCDet_HOME>/data/kitti/kitti_infos_val.pkl

import argparse
import pickle

from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval


def _stats(pred_infos, gt_infos):

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