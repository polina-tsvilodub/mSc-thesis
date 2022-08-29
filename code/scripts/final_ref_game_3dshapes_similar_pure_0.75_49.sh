#!/bin/sh
for i in 0.75
do
    for j in "pure" #"topk_temperature" # "greedy" # 
    do
        for sp in "../../data/final/models/final_pretrained_speaker_3dshapes_exh_49vocab_tf_desc05_padding_pureDecoding_cont4__3.pkl" # "models/decoder-coco-512dim-teacher_forcing_scheduled_desc_05_byEp-7.pkl" #
        do
            cd ../src && python ../src/reference_game_train.py -ep=2 -ex="3dshapes"\
            -dv="train_logs/val_img_IDs_unique_3dshapes_randomSample10000_list_str.pt" -ms=15 -d="train_logs/ref_game_img_IDs_unique_imgIDs4train_3dshapes.pt"\
            -b=64 -l="../../data/final/reference_games/3dshapes/ref_game_3dshapes_pure_decoding_Ls075_49vocab_similar_test.txt" -vf="../../data/vocab3dshapes_fixed.pkl" -tl="../../data/final/reference_games/3dshapes/losses_ref_game_3dshapes_pure_decoding_Ls075_49vocab_similar_test_"\
            -tm="../../data/final/reference_games/3dshapes/metrics_ref_game_3dshapes_pure_decoding_Ls075_49vocab_similar_test_" -vl="" -vm=""\
            -s_pre=$sp -n_img=0 -p="similar" -l_s=$i -str=$j -entr=0.1 -l_t="joint"
        done
    done
done 