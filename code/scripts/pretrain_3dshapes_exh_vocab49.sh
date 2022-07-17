#!/bin/sh
cd ../src && python ../src/pretrain_speaker.py -ep=5 -ex="3dshapes" -d="" -dv="train_logs/val_img_IDs_unique_3dshapes.pt"\
    -ms=25 -b=64 -l='../../data/pretraining_speaker_3dshapes_exh_49vocab_teacher_forcing05_desc_pureDecoding_padding_indFixed.txt'\
    -vf="../../data/vocab3dshapes_fixed.pkl" -tl="../../data/final/losses_final_pretrained_speaker_3dshapes_exh_49vocab_tf_desc05_padding_pureDecoding_"\
    -tm="../../data/final/metrics_final_pretrained_speaker_3dshapes_exh_49vocab_tf_desc05_padding_pureDecoding_"\
    -vm="../../data/final/val_final_pretrained_speaker_3dshapes_exh_49vocab_tf_desc05_padding_pureDecoding_"\
    -n_img=320 -wp='../../data/final/models/final_pretrained_speaker_3dshapes_exh_49vocab_tf_desc05_padding_pureDecoding_'