#!/bin/sh
for i in 0.75
do
    for j in "pure" #"topk_temperature" # "greedy" # 
    do
        for sp in "../../data/final/models/final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_cont_4__3.pkl" # "models/decoder-coco-512dim-teacher_forcing_scheduled_desc_05_byEp-7.pkl" #
        do
            cd ../src && python ../src/reference_game_train.py -ep=2 -ex="coco"\
            -dv="notebooks/val_split_IDs_from_COCO_train_tensor.pt" -ms=15\
            -b=64 -l="../../data/final/reference_games/coco/ref_game_coco_pure_decoding_Ls075_4000vocab_random.txt" -vf="../../data/vocab4000.pkl" -tl="../../data/final/reference_games/coco/losses_ref_game_coco_pure_decoding_Ls075_4000vocab_random_"\
            -tm="../../data/final/reference_games/coco/metrics_ref_game_coco_pure_decoding_Ls075_4000vocab_random_" -vl="" -vm=""\
            -s_pre=$sp -n_img=0 -p="random" -l_s=$i -str=$j -entr=0.1
        done
    done
done 