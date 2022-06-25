#!/bin/sh
for i in 0.1, 0.2
do
    python reference_game_train.py -ep=1 -ex="coco" -d="" -ms=15 -b=64 -l="../../data/sacred_ref_game_coco_test.txt" -vf="../../data/vocab4000.pkl" -tl="" -tm="" -vl="" -vm="" -sd="" -le="" -ld="" -s_pre="" -n_img="" -p="random" -l_s=i -str="pure" -entr=0.1
done    