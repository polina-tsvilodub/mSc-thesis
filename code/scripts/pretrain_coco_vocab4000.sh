#!/bin/sh
cd ../src && python ../src/pretrain_speaker.py -pre_s -sw="../../data/final/models/final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_cont__1.pkl"\
    -ep=2 -ex="coco" -d="" -dv="notebooks/val_split_IDs_from_COCO_train_tensor.pt"\
    -ms=15 -b=64 -l='../../data/pretraining_speaker_noEnc_prepend_512dim_4000vocab_teacher_forcing05_desc_pureDecoding_padding_indFixed_cont2.txt'\
    -vf="../../data/vocab4000.pkl" -tl="../../data/final/losses_final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_cont_2_"\
    -tm="../../data/final/metrics_final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_cont_2_"\
    -vm="../../data/final/val_final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_cont_2_"\
    -n_img=320 -wp='../../data/final/models/final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_cont_2_'