#!/bin/bash

python3 ./train.py -s -l --no_mix_epoch=1 --batch_size=128 --epochs=300 --name=model_3.pth --weight_decay=6e-2 --max_lr=55e-5
echo "Training completed."