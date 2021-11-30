#-*-coding:utf-8-*-
"""
    CCT Training Module
"""
import os
import torch
import argparse

import torchvision

from py.CCT import CCT

from torch import optim
from torchvision import transforms

from py.LabelSmoothing import LabelSmoothingCrossEntropy
from py.train_utils import *

load_path = ""
default_chkpt_path = "./check_points/"
default_model_path = "./model/"

# Calculate accurarcy (correct prediction counter)
def accCounter(pred:torch.FloatTensor, truth:torch.FloatTensor)->int:
    _, max_pos = torch.max(pred, dim = 1)
    return torch.sum(max_pos == truth)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 20, help = "Training lasts for . epochs")
    parser.add_argument("--batch_size", type = int, default = 100, help = "Batch size for range image packs.")
    parser.add_argument("--eval_time", type = int, default = 5, help = "Eval every <eval_time> batches.")
    parser.add_argument("--chkpt_ntv", type = int, default = 50, help = "Interval for checkpoints.")
    parser.add_argument("--name", type = str, default = "model_1.pth", help = "Model name for loading")
    parser.add_argument("--weight_decay", type = float, default = 3e-2, help = "Weight Decay in AdamW")
    parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-c", "--cuda", default = False, action = "store_true", help = "Use CUDA to speed up training")
    parser.add_argument("-l", "--load", default = False, action = "store_true", help = "Load checkpoint or trained model.")
    parser.add_argument("-o", "--optimize", default = False, action = "store_true", help = "Optimization, small lr.")
    args = parser.parse_args()

    use_cuda            = args.cuda
    epochs              = args.epochs
    batch_size          = args.batch_size
    eval_time           = args.eval_time
    del_dir             = args.del_dir
    use_cuda            = args.cuda
    chkpt_ntv           = args.chkpt_ntv
    use_load            = args.load
    load_path           = default_model_path + args.name
    
    aug_to_tensor = transforms.Compose([
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.1),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean = [.5, .5, .5], std = [.5, .5, .5]),
    ])

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [.5, .5, .5], std = [.5, .5, .5]),
    ])

    device = torch.device(type = 'cpu')
    if use_cuda and torch.cuda.is_available():
        device = torch.device(0)
        print("Cuda is operationing")
    else:
        use_cuda = False
        print("CUDA not available.")
    
    writer = getSummaryWriter(epochs, del_dir)

    train_set = getCIFAR10Dataset(True, aug_to_tensor, batch_size)
    test_set = getCIFAR10Dataset(False, to_tensor, batch_size)
    
    model:CCT = CCT()       # default parameters
    model.to(device)
    if use_load == True and os.path.exists(load_path):
        model.loadFromFile(load_path)
    else:
        print("Not loading or load path '%s' does not exist."%(load_path))
    loss_func = LabelSmoothingCrossEntropy(0.1)
    batch_num = len(train_set)

    opt = optim.AdamW(model.parameters(), lr = 1.0, betas = (0.9, 0.999), weight_decay=args.weight_decay)
    # opt_sch = optim.lr_scheduler.MultiStepLR(opt, [5 * batch_num, 10 * batch_num, 15 * batch_num, 20 * batch_num], gamma = 0.1, last_epoch = -1)
    # opt_sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0 = 5, T_mult = 2, eta_min = 5e-8, last_epoch = -1)
    opt_sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr, last_epoch=-1)
    train_cnt = 0
    for ep in range(epochs):
        model.train()
        train_acc_cnt = 0
        train_num = 0
        for i, (px, py) in enumerate(train_set):
            px:torch.Tensor = px.to(device)
            py:torch.Tensor = py.to(device)
            pred = model(px)
            # one_hot:torch.Tensor = makeOneHot(py, device)
            loss = loss_func(pred, py)
            train_acc_cnt += accCounter(pred, py)
            train_num += len(pred)
            opt.zero_grad()
            loss.backward()
            opt.step()
            opt_sch.step()
            if train_cnt % eval_time == 1:
                model.eval()
                with torch.no_grad():
                    ## wocao not optimizing at all, fuck
                    ## +++++++++++ Load from Test set ++++++++=
                    test_length = len(test_set) * batch_size
                    test_acc_cnt = 0
                    test_loss = torch.zeros(1).to(device)
                    for ptx, pty in test_set:
                        ptx:torch.Tensor = ptx.to(device)
                        pty:torch.Tensor = pty.to(device)
                        pred = model(ptx)
                        test_acc_cnt += accCounter(pred, pty)
                        # one_hot_test:torch.Tensor = makeOneHot(pty, device)
                        test_loss += loss_func(pred, pty)
                    train_acc = train_acc_cnt / train_num
                    test_acc = test_acc_cnt / test_length
                    print("Epoch: %4d / %4d\t Batch %4d / %4d\t train loss: %.4f\t test loss: %.4f\t acc: %.4f\t test acc: %.4f\t lr: %f"%(
                            ep, epochs, i, batch_num, loss.item(), test_loss.item(), train_acc, test_acc, opt_sch.get_last_lr()[-1]
                    ))
                    train_num = 0
                    train_acc_cnt = 0
                    writer.add_scalar('Loss/Train Loss', loss, train_cnt)
                    writer.add_scalar('Loss/Test loss', test_loss, train_cnt)
                    writer.add_scalar('Acc/Train Set Accuracy', train_acc, train_cnt)
                    writer.add_scalar('Acc/Test Set Accuracy', test_acc, train_cnt)
                model.train()
            if train_cnt % chkpt_ntv == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': opt.state_dict()},
                    "%schkpt_%d.pt"%(default_chkpt_path, train_cnt)
                )
            train_cnt += 1
    torch.save({
        'model': model.state_dict(),
        'optimizer': opt.state_dict()},
        "%s/model_1.pth"%(default_model_path)
    )
    writer.close()
    print("Output completed.")
    