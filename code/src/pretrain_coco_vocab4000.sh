#!/bin/sh
python pretrain_speaker.py -ep=5 -ex="coco" -d="" -dv="notebooks/val_split_IDs_from_COCO_train_tensor.pt"\
    -ms=15 -b=64 -l='../../data/pretraining_speaker_noEnc_prepend_512dim_4000vocab_teacher_forcing05_desc_pureDecoding_padding_indFixed.txt'\
    -vf="../../data/vocab4000.pkl" -tl="../../data/final/losses_final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_"\
    -tm="../../data/final/metrics_final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_"\
    -vm="../../data/final/val_final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_"\
    -n_img=320 -wp='../../data/final/models/final_pretrained_speaker_coco_4000vocab_tf_desc05_padding_pureDecoding_'