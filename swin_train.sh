#!/bin/bash

python3 ./swin.py -s -l --batch_size=40 --epochs=250 --name=model_1.pth --weight_decay=6e-2 --max_lr=55e-5
echo "Training completed."