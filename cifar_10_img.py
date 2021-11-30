"""
    convert CIFAR_10 RAW Data to CIFAR_10 img (png)
"""

import numpy as np
from tqdm import tqdm
from py.train_utils import *

if __name__ == "__main__":
    train_set = CIFAR10Images(True)
    test_set = CIFAR10Images(False)
    train_folder_cnt = np.ones(10, dtype=int) 
    test_folder_cnt = np.ones(10, dtype=int) 
    img_path = "./imgs/"
    train_len, test_len = len(train_set), len(test_set)
    for i in tqdm(range(train_len), desc='Processing'):
        px, py = train_set[i]
        pic_cnt = train_folder_cnt[py]
        save_path = "%strain/%d/%d.png"%(img_path, py, pic_cnt)
        px.save(save_path)
        train_folder_cnt[py] += 1
    for i in tqdm(range(test_len), desc='Processing'):
        px, py = test_set[i]
        pic_cnt = test_folder_cnt[py]
        save_path = "%stest/%d/%d.png"%(img_path, py, pic_cnt)
        px.save(save_path)
        test_folder_cnt[py] += 1
    print("Train counter sum: %d, test counter sum: %d"%(sum(train_folder_cnt), sum(test_folder_cnt)))