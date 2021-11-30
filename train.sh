#!/bin/bash

python3 ./train.py -c --batch_size=64 --epochs=20 --name=chkpt_7750.pt -l
echo "Training completed."