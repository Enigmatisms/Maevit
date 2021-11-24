#-*-coding:utf-8-*-
"""
    CCT Training Module
"""
import os
import torch
import shutil
import argparse
from datetime import datetime
from torch.autograd import Variable as Var
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10

from CCT import CCT

from torch import optim
from torch import nn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from train_utils import *

load_path = ""
default_chkpt_path = "./check_points/"
default_model_path = "./model/"

# Calculate accurarcy (correct prediction counter)
def accCounter(pred:torch.FloatTensor, truth:torch.FloatTensor)->int:
    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 40, help = "Training lasts for . epochs")
    parser.add_argument("--batch_size", type = int, default = 5, help = "Batch size for range image packs.")
    parser.add_argument("--eval_time", type = int, default = 20, help = "Eval every <eval_time> batches.")
    parser.add_argument("--chkpt_ntv", type = int, default = 400, help = "Interval for checkpoints.")
    parser.add_argument("--name", type = str, default = "model_1.pth", help = "Model name for loading")
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

    to_tensor = transforms.ToTensor()
    
    if args.load:
        if os.path.exists(load_path) == False:
            print("No such file as '%s'! Model failed to load."%(load_path))
        else:
            pass
            # model.loadFromFile(load_path)
    device = torch.device(type = 'cpu')
    if use_cuda and torch.cuda.is_available():
        device = torch.device(0)
    else:
        use_cuda = False
        print("CUDA not available.")
    
    writer = getSummaryWriter(epochs, del_dir)

    train_set = getCIFAR10Dataset(True, to_tensor, batch_size)
    test_set = getCIFAR10Dataset(False, to_tensor, batch_size)
    
    # ------------ model --------------
    # TODO: ??import???????
    model:CCT = CCT()       # default parameters
    if use_load == True and os.path.exists(load_path):
        model.loadFromFile(load_path)
    else:
        print("Not loading or load path '%s' does not exist."%(load_path))
    
    loss_func = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.paramters(), lr = 1e-3)
    opt_sch = optim.lr_scheduler.MultiStepLR([15, 30, 45, 60], gamma = 0.1, last_epoch = -1)

    batch_num = len(train_set) // batch_size
    for ep in range(epochs):
        model.train()
        train_cnt = 0
        train_acc_cnt = 0
        for i, px, py in enumerate(train_set):
            train_cnt += 1
            loss = torch.zeros(1).to(device)
            px:torch.Tensor = px.to(device)
            py:torch.Tensor = py.to(device)
            pred = model(px)
            loss = loss_func(pred, py)
            train_acc_cnt += accCounter(pred, py)
            if train_cnt % eval_time == 1:
                with torch.no_grad():
                    model.eval()
                    ## +++++++++++ Load from Test set ++++++++=
                    test_length = len(test_set)
                    test_acc_cnt = 0
                    test_loss = torch.zeros(1).to(device)
                    for ptx, pty in test_set:
                        ptx:torch.Tensor = ptx.to(device)
                        pty:torch.Tensor = pty.to(device)
                        pred = model(ptx)
                        test_acc_cnt += accCounter(pred, pty)
                        test_loss += loss_func(pred, pty)
                    train_acc = train_acc_cnt / train_cnt
                    test_acc = test_acc_cnt / test_length
                    print("Epoch: %3d / %3d\t Batch %4d / %4d\t train loss: %.4f\t test loss: %.4f\t acc: %.4f\t test acc: %.4f\t lr: %f"%(
                            ep, epochs, i, batch_num, loss.item(), test_loss.item(), train_acc, test_acc, opt_sch.param_groups[0]['lr']
                    ))
                    train_cnt = 0
                    # writer.add_scalar('Loss/Train Loss', loss, train_cnt)
                    # writer.add_scalar('Loss/Test loss', test_loss, train_cnt)
                    # writer.add_scalar('Acc/Train Set Accuracy', train_acc, train_cnt)
                    # writer.add_scalar('Acc/Test Set Accuracy', test_acc, train_cnt)
            if train_cnt % chkpt_ntv == 0:
                # torch.save({
                #     'model': model.state_dict(),
                #     'optimizer': opt.state_dict()},
                #     "%schkpt_%d.pt"%(default_chkpt_path, train_cnt)
                # )
                pass
            
        # For recontruction task:
        # save_image(gen.detach().clamp_(0, 1), "..\\imgs\\G_%d.jpg"%(epoch + 1), 1)
    # torch.save({
    #     'model': model.state_dict(),
    #     'optimizer': opt.state_dict()},
    #     "%s/model_1.pth"%(default_model_path)
    # )
    writer.close()
    print("Output completed.")
    