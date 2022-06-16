#!/usr/bin/python3
# python calc_mean_mAP.py --save_to_file --exp_names disabled_st_bs4_trial1_818cd7c disabled_st_bs4_trial2_818cd7c disabled_st_bs4_trial3_818cd7c

import argparse
import os
import pickle
import re

import numpy as np



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--exp_names', required=True, nargs=argparse.REMAINDER,
                        help='--exp_names <test-name-1>, <test-name-2> ..')
    parser.add_argument('--thresh', type=str, default='0.5, 0.25, 0.25')
    parser.add_argument('--save_to_file', action='store_true', default=True, help='')
    parser.add_argument('--result_tag', type=str, default=None, help='extra tag for this experiment')
    args = parser.parse_args()
    return args


def get_sorted_text_files(dirpath):
    a = [s for s in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, s)) and s.endswith('.txt')]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    return a



def calc_mean_mAP():
    """
    Takes n experiments and calculate mean of max class mAP
    """
    args = parse_config()
    #THRESH_ = [float(x) for x in args.thresh.split(',')]
    assert args.exp_names is not None
    exp_names = [str(x) for x in args.exp_names]    
    # metric=""
    # if THRESH_[0]==0.5:metric.join("Car AP_R40@0.70, 0.50, 0.50")
    # if THRESH_[1]==0.25:metric.join("Pedestrian AP_R40@0.50, 0.25, 0.25")
    # if THRESH_[2]==0.25:metric.join("Cyclist AP_R40@0.50, 0.25, 0.25")

    metric = ["Car AP_R40@0.70, 0.70, 0.70","Pedestrian AP_R40@0.50, 0.50, 0.50","Cyclist AP_R40@0.50, 0.50, 0.50"]
    pattern = re.compile(r'({0})'.format('|'.join(metric)))
    max_results=[]
    eval_list=None

    print("\n#--------------------Calculate Mean mAP-----------------------#\n")
    print("\nDefined Metric")
    for m in metric:
        print(m)
    print("\nExperiment(s))")
    for e in exp_names:
        print(e)

    if args.save_to_file:
        res_text_file= os.path.join(os.getcwd(), "{}_results.txt".format(exp_names[0] if args.result_tag is None else exp_names[0] + args.result_tag))
        fw=open(res_text_file, 'w')
        fw.write("\n#--------------------Calculate Mean mAP-----------------------#\n")
        fw.write("\nDefined Metric\n")
        fw.write(str(metric))
        fw.write("\nExperiment(s)\n")
        fw.write(str(exp_names))

    for _exp in exp_names:
        curr_eval_list_file = os.path.join("output/cfgs/kitti_models/pv_rcnn_ssl_60", _exp, "eval/eval_with_train/eval_list_val.txt")
        if eval_list is None and os.path.isfile(curr_eval_list_file):
            with open(curr_eval_list_file) as f_eval:
                eval_list = list(set(map(int, f_eval.readlines())))# take only unique entries 
                print("\nEvaluated Epochs")
                print(*[str(i) for i in eval_list], sep=",")
                if args.save_to_file:
                    fw.write("\nEvaluated Epochs")
                    fw.write(str(eval_list))

        
        
        curr_res_dir = os.path.join("output/cfgs/kitti_models/pv_rcnn_ssl_60", _exp)
        if not os.path.isdir(curr_res_dir): 
            continue

        text_files = get_sorted_text_files(curr_res_dir)
        if len(text_files)==0:
            print("No text file found containing results")
            continue

        selected_file=os.path.join(curr_res_dir, text_files[0])
        print("\nScanning {} for evaluated results\n".format(selected_file))# can be filtered based on date-time
        if args.save_to_file: 
            fw.write("\nScanning {} for evaluated results\n".format(selected_file))
        
        # get data from file 
        eval_results=[]
        line_numbers=[]
        linenum = 0

        with open(selected_file) as fp:
            for line in fp:
                linenum += 1
                if pattern.search(line) != None: # If a match is found 
                    line_numbers.append(linenum+3) # add following res-line-number into list
                if linenum in line_numbers:
                    res_=np.fromstring( line.strip().split("3d   AP:")[1], dtype=np.float64, sep=',' )
                    #print(res_)
                    eval_results.append(res_)
        
        # reshape records based on eval_list
        eval_results=np.array(eval_results).reshape(len(eval_list),-1)
        print("\nmAP(s)")
        print(*[str(np.round_(i, decimals=2)) for i in eval_results], sep="\n")
        if args.save_to_file:
            fw.write("\nmAP(s)")
            fw.write(str(np.round_(eval_results, decimals=2)))
        
        current_max=np.max(eval_results, axis=0)
        max_results.append(current_max)
        print("\nMax mAP")
        print(*[str(np.round_(i, decimals=2)) for i in current_max], sep=", ")
        if args.save_to_file: 
            fw.write("\nMax mAP")
            fw.write(str(np.round_(current_max, decimals=2)))
        print("\n\n")

    print("\n\n----------------Final Results----------------\n\n")
    # all results have been added
    max_results=np.array(max_results)
    print("Max mAP(s)\n")
    print(*[str(np.round_(i, decimals=2)) for i in max_results], sep="\n")

    if args.save_to_file: 
        fw.write("\n\n----------------Final Results----------------\n\n")
        fw.write("Max mAP(s)\n")
        fw.write(str(np.round_(max_results, decimals=2)))

    mean_res=np.mean(max_results, axis=0)
    print("\nMean mAP")
    print(*[str(np.round_(i, decimals=2)) for i in mean_res], sep=", ")
    if args.save_to_file: 
        fw.write("\nMean mAP")
        fw.write(str(np.round_(mean_res, decimals=2)))

if __name__ == "__main__":
    calc_mean_mAP()
