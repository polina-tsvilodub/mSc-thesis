#!/bin/sh
for i in 0
do
    python reference_game_train.py -ep=1 -ex="coco" -d="train_logs/ref-game_img_IDs_15000_coco_lf01.pt"\
    -dv="notebooks/val_split_IDs_from_COCO_train_tensor.pt" -ms=15\
    -b=64 -l="../../data/sacred_ref_game_coco_Lf_1_.txt" -vf="../../data/vocab4000.pkl" -tl="reference_game_coco_losses_tracked_Lf1.csv"\
    -tm="reference_game_coco_metrics_tracked_Lf1.csv" -vl="" -vm="" -sd="" -le="" -ld=""\
    -s_pre="models/decoder-noEnc-prepend-512dim-4000vocab-rs1234-wEmb-cont-7.pkl" -n_img=0 -p="random" -l_s=$i -str="pure" -entr=0.1
done 