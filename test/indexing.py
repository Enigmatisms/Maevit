
import torch

def getIndex(win_size:int) -> torch.LongTensor:
    ys, xs = torch.meshgrid(torch.arange(win_size), torch.arange(win_size), indexing = 'ij')
    coords = torch.cat((ys.unsqueeze(dim = -1), xs.unsqueeze(dim = -1)), dim = -1).view(-1, 2)
    diff = coords[None, :, :] - coords[:, None, :]          # interesting broadcasting, needs notes
    diff += win_size - 1
    index = diff[:, :, 0] * (2 * win_size - 1) + diff[:, :, 1]
    return index

if __name__ == "__main__":
    print(getIndex(3))