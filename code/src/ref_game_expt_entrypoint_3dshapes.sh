#!/bin/sh
for i in 0 0.75
do
    for j in "pure" "greedy" 
    do
        for sp in "models/decoder-3dshapes-512dim-49vocab-rs1234-exh-3.pkl" # "models/decoder-noEnc-prepend-512dim-4000vocab-rs1234-wEmb-cont-7.pkl" 
        do
            python reference_game_train.py -ep=1 -ex="3dshapes" -d="train_logs/ref_game_img_IDs_unique_imgIDs4train_3dshapes.pt"\
            -dv="train_logs/val_img_IDs_unique_3dshapes.pt" -ms=25\
            -b=64 -l="../../data/sacred_ref_game_3dshapes_wPure_decoding_tracked.txt" -vf="../../data/vocab3dshapes_fixed.pkl" -tl="reference_game_3dshapes_losses_tracked_wPure_decoding_LFonly_"\
            -tm="reference_game_3dshapes_metrics_tracked_wPure_decoding_LFonly_" -vl="" -vm=""\
            -s_pre=$sp -n_img=15000 -p="random" -l_s=$i -str=$j -entr=0.1
        done
    done
done 