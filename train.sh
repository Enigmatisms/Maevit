#!/bin/bash

python3 ./train.py -c -l --no_aug_epoch=320 --batch_size=128 --epochs=300 --name=model_2.pth --weight_decay=6e-2
echo "Training completed."