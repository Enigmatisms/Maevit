import torch
from timm.data import create_dataset, create_loader

if __name__ == "__main__":
    data = create_dataset("Imagenette", "../../dataset/imagenette2-320/", "train", is_training = True)
    loader = create_loader(data, 224, 50, True)
    cnt = 0 
    print(len(loader))
    for x, y in loader:
        print(x.shape, y.shape)
        cnt += 1
        if cnt > 2000:
            break