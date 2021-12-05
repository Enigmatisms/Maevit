#!/bin/bash

python3 ./train.py -l -s --no_aug_epoch=450 --batch_size=128 --epochs=500 --name=chkpt_143200.pt --weight_decay=6e-2 --max_lr=55e-5
echo "Training completed."