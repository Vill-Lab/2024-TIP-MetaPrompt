#!/bin/bash

DATA=./data
TRAINER=Meta_B2N
SHOTS=16
LOADEP=8
CFG=vit_b16
DATASET=$1

for SEED in 1 2 3
do 

    COMMON_DIR=${DATASET}/${TRAINER}/${CFG}_shots_${SHOTS}/seed${SEED}
    DIRTRAIN=./B2N/train_base/${COMMON_DIR}
    DIRTEST=./B2N/test_new/${COMMON_DIR}

    if [ -d "$DIRTRAIN" ]; then
        echo "Oops! The results exist at ${DIRTRAIN} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/b2n/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIRTRAIN} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES base \
        TRAINER.META.PREC 'fp16'
    fi

    if [ -d "$DIRTEST" ]; then
        echo "Oops! The results exist at ${DIRTEST} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/b2n/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIRTEST} \
        --model-dir ${DIRTRAIN} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES new
    fi

done