#-*-coding:utf-8-*-
"""
    This is a template for torch main.py
"""
import os
import torch
import shutil
import argparse
from datetime import datetime
from torch.autograd import Variable as Var
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10
from PIL import Image

from torch import optim
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

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
    
    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    writer = SummaryWriter(log_dir = logdir+time_stamp)
    download = (len(os.listdir("../dataset/")) == 0)
    train_set = DataLoader(
        CIFAR10("../dataset/", 
            train = True, download = download, transform = to_tensor),
        batch_size = batch_size, shuffle = True,
    )
    
    test_set = DataLoader(
        CIFAR10("../dataset/", 
            train = False, download = False, transform = to_tensor),
        batch_size = batch_size, shuffle = True,
    )
    
    # loss_func = nn.MSELoss()
    # opt = optim.AdamW(model.paramters(), lr = 1e-3)
    # opt_sch = optim.lr_scheduler.MultiStepLR([1, 3, 5, 10], gamma = 0.1, last_epoch = -1)

    batch_num = 0 # batch_num = len(dataset) // batch_size
    train_cnt = 0
    for ep in range(epochs):
        # model.train()
        for i, px, py in enumerate(train_set):
            if use_cuda:
                px = px.to(device)
                py = py.to(device)
            ## +++++++ Load from Dataset iterates +++++++ 
            ## +++++++++++++++ Training +++++++++++++++++
            
            with torch.no_grad():
                # model.eval()
                ## +++++++++++ Load from Test set ++++++++=

                # print("Epoch: %3d / %3d\t Batch %4d / %4d\t train loss: %.4f\t test loss: %.4f\t acc: %.4f\t test acc: %.4f\t lr: %f"%(
                #         ep, epochs, i, batch_num, loss.item(), test_loss.item(), train_acc, test_acc, opt_sch.param_groups[0]['lr']
                # ))
                # writer.add_scalar('Loss/Train Loss', loss, train_cnt)
                # writer.add_scalar('Loss/Test loss', test_loss, train_cnt)
                # writer.add_scalar('Acc/Train Set Accuracy', train_acc, train_cnt)
                # writer.add_scalar('Acc/Test Set Accuracy', test_acc, train_cnt)
                pass
            if train_cnt % chkpt_ntv == 0:
                # torch.save({
                #     'model': model.state_dict(),
                #     'optimizer': opt.state_dict()},
                #     "%schkpt_%d.pt"%(default_chkpt_path, train_cnt)
                # )
                pass
            train_cnt += 1
            
        # For recontruction task:
        # save_image(gen.detach().clamp_(0, 1), "..\\imgs\\G_%d.jpg"%(epoch + 1), 1)
    # torch.save({
    #     'model': model.state_dict(),
    #     'optimizer': opt.state_dict()},
    #     "%s/model_1.pth"%(default_model_path)
    # )
    writer.close()
    print("Output completed.")
    