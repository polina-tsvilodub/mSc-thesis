#!/bin/sh
for i in 0.75 1
do
    for j in "exp" #"topk_temperature" # "greedy" # 
    do
        for sp in "models/decoder-noEnc-prepend-512dim-4000vocab-rs1234-wEmb-cont-7.pkl" # "models/decoder-coco-512dim-teacher_forcing_scheduled_desc_05_byEp-7.pkl" #
        do
            cd ../src && python ../src/reference_game_train.py -ep=1 -ex="coco" -d="train_logs/ref-game_img_IDs_15000_coco_lf01.pt"\
            -dv="notebooks/val_split_IDs_from_COCO_train_tensor.pt" -ms=15\
            -b=64 -l="../../data/sacred_ref_game_coco_exp_decoding_fixed_tracked.txt" -vf="../../data/vocab4000.pkl" -tl="reference_game_coco_losses_tracked_exp_decoding_fixed_base_speaker_"\
            -tm="reference_game_coco_metrics_tracked_exp_decoding_fixed_base_speaker_" -vl="" -vm=""\
            -s_pre=$sp -n_img=0 -p="random" -l_s=$i -str=$j -entr=0.1
        done
    done
done 