from py.train_utils import getCIFAR10Dataset
from torchvision import transforms
import torch

tf = transforms.ToTensor()

def makeOneHot(labels:torch.Tensor, device)->torch.Tensor:
    dtype = labels.type()
    length = labels.shape[0]
    one_hot = torch.zeros(length, 10).to(device)
    one_hot[torch.arange(length), labels] = 1.0
    return one_hot

if __name__ == "__main__":
    train = getCIFAR10Dataset(True, tf, 50)
    test = getCIFAR10Dataset(True, tf, 50)
    device = torch.device(0)
    for i, (x, y), in enumerate(train):
        print(x.shape)
        print(y)
        print(makeOneHot(y, device))
        break