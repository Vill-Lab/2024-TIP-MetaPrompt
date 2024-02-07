#!/bin/bash

DATA=./data
TRAINER=Meta_DG
CFG=vit_b16

DATASET=$1
SHOTS=$2

if [ "$DATASET" == "domainnet" ]; then
    N=6
else
    N=4
fi


for ((TGT=0;TGT<$N;TGT++))
do
    for SEED in 1 2 3
    do
        DIR=./DG/${DATASET}/${TRAINER}/${CFG}_shots_${SHOTS}/${TGT}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            CUDA_VISIBLE_DEVICES=0 python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.ID ${TGT}
        fi
    done
done