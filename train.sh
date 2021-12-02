#!/bin/bash

python3 ./train.py -c --batch_size=128 --epochs=135 --name=model_2.pth -l --weight_decay=6e-2
echo "Training completed."