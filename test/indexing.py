
import torch
import einops

def getIndex(win_size:int) -> torch.LongTensor:
    ys, xs = torch.meshgrid(torch.arange(win_size), torch.arange(win_size), indexing = 'ij')
    coords = torch.cat((ys.unsqueeze(dim = -1), xs.unsqueeze(dim = -1)), dim = -1).view(-1, 2)
    diff = coords[None, :, :] - coords[:, None, :]          # interesting broadcasting, needs notes
    diff += win_size - 1
    index = diff[:, :, 0] * (2 * win_size - 1) + diff[:, :, 1]
    return index

# Patch Merging Einops is verified
def mergePatchTest(X:torch.Tensor):
    return einops.rearrange(X, 'N wn (H m1) (W m2) C -> N wn H W (m1 m2 C)', m1 = 2, m2 = 2)

# Patch partition Einops is verified
def paritionTest(X:torch.Tensor):
    return einops.rearrange(X, 'N (m1 H) (m2 W) C -> N (m1 m2) (H W) C', m1 = 3, m2 = 3)

if __name__ == "__main__":
    print(getIndex(7), "\n")
    # # A = torch.arange(128).view(2, 2, 4, 4, 2)
    # # print(A)
    # # print(mergePatchTest(A))
    # # print("\n")
    # B = torch.arange(36).view(1, 6, 6, 1)
    # print(B)
    # print(paritionTest(B))