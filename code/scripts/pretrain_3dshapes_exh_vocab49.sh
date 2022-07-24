#!/bin/sh
cd ../src && python ../src/pretrain_speaker.py -ep=3 -ex="3dshapes" -pre_s -sw="../../data/final/models/final_pretrained_speaker_3dshapes_exh_49vocab_tf_desc05_padding_pureDecoding_cont3__2.pkl"\
    -d="" -dv="train_logs/val_img_IDs_unique_3dshapes.pt"\
    -ms=25 -b=64 -l='../../data/pretraining_speaker_3dshapes_exh_49vocab_teacher_forcing05_desc_pureDecoding_padding_indFixed_cont4.txt'\
    -vf="../../data/vocab3dshapes_fixed.pkl" -tl="../../data/final/losses_final_pretrained_speaker_3dshapes_exh_49vocab_tf_desc05_padding_pureDecoding_cont4_"\
    -tm="../../data/final/metrics_final_pretrained_speaker_3dshapes_exh_49vocab_tf_desc05_padding_pureDecoding_cont4_"\
    -vm="../../data/final/val_final_pretrained_speaker_3dshapes_exh_49vocab_tf_desc05_padding_pureDecoding_cont4_"\
    -n_img=64 -wp='../../data/final/models/final_pretrained_speaker_3dshapes_exh_49vocab_tf_desc05_padding_pureDecoding_cont4_'