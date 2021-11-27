#!/bin/bash

python3 ./train.py -c --batch_size=64 --epochs=100 --name=chkpt_800.pt
echo "Training completed."