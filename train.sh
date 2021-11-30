#!/bin/bash

python3 ./train.py -c --batch_size=64 --epochs=32 --name=model_2.pth -l
echo "Training completed."