#-*-coding:utf-8-*-
"""
    CCT Training Module
"""
import os
import torch
import argparse
from functools import partial

from torch import optim
from torch.cuda import amp

from py.CCT import CCT
from py.train_utils import *
from py.LabelSmoothing import LabelSmoothingCrossEntropy
from py.LECosineAnnealing import LECosineAnnealingSmoothRestart
from timm.utils import NativeScaler
from timm.models import model_parameters
from timm.scheduler import CosineLRScheduler

default_chkpt_path = "./check_points/"
default_model_path = "./model/"

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type = int, default = 20, help = "Training lasts for . epochs")
parser.add_argument("--batch_size", type = int, default = 100, help = "Batch size for range image packs.")
parser.add_argument("--eval_time", type = int, default = 50, help = "Eval every <eval_time> batches.")
parser.add_argument("--cooldown_epoch", type = int, default = 10, help = "Epochs to cool down lr after cyclic lr.")
parser.add_argument("--no_aug_epoch", type = int, default = 320, help = "The epoch when data augmentation is cancelled")
parser.add_argument("--eval_div", type = int, default = 5, help = "Output every <...> times in evaluation")
parser.add_argument("--name", type = str, default = "model_1.pth", help = "Model name for loading")
parser.add_argument("--weight_decay", type = float, default = 3e-2, help = "Weight Decay in AdamW")
parser.add_argument("--max_lr", type = float, default = 55e-5, help = "Max learning rate")
parser.add_argument("--min_lr", type = float, default = 1e-5, help = "Min learning rate")
parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
parser.add_argument("-l", "--load", default = False, action = "store_true", help = "Load checkpoint or trained model.")
parser.add_argument("-o", "--optimize", default = False, action = "store_true", help = "Optimization, small lr.")
parser.add_argument("-s", "--use_scaler", default = False, action = "store_true", help = "Use AMP scaler to speed up")
args = parser.parse_args()

# Calculate accurarcy (correct prediction counter)
def accCounter(pred:torch.FloatTensor, truth:torch.FloatTensor)->int:
    _, max_pos = torch.max(pred, dim = 1)
    return torch.sum(max_pos == truth)

def get_sch_lr(sch:CosineLRScheduler, start_lr:float, i:int)->float:
    return sch.get_epoch_values(i)[0] * start_lr

def main():
    epochs              = args.epochs
    batch_size          = args.batch_size
    eval_time           = args.eval_time
    del_dir             = args.del_dir
    use_load            = args.load
    use_amp             = args.use_scaler
    load_path           = default_model_path + args.name
    eval_div            = args.eval_div

    train_set = getCIFAR10Dataset(True, True, batch_size)
    train_set_no_aug = getCIFAR10Dataset(True, False, batch_size)
    test_set = getCIFAR10Dataset(False, False, 100)

    device = None
    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(-1)
    device = torch.device(0)
    
    model:CCT = CCT().cuda()       # default parameters
    if use_load == True and os.path.exists(load_path):
        model.loadFromFile(load_path)
    else:
        print("Not loading or load path '%s' does not exist."%(load_path))
    loss_func = LabelSmoothingCrossEntropy(0.1).cuda()
    batch_num = len(train_set)

    augment_cancelled = False

    opt = optim.AdamW(model.parameters(), lr = 1.0, betas = (0.9, 0.999), weight_decay=args.weight_decay)
    # lec_sch_func = LECosineAnnealingSmoothRestart(55e-5, 1e-4, 1e-4, 1e-5, epochs * batch_num, 64, True)
    min_max_ratio = args.min_lr / args.max_lr
    lec_sch_func = CosineLRScheduler(opt, t_initial = epochs // 2, t_mul = 1, lr_min = min_max_ratio, decay_rate = 0.1,
            warmup_lr_init = min_max_ratio, warmup_t = 10, cycle_limit = 2, t_in_epochs = True)
    epochs = lec_sch_func.get_cycle_length() + args.cooldown_epoch
    writer = getSummaryWriter(epochs, del_dir)

    opt_sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda = partial(get_sch_lr, lec_sch_func, args.max_lr), last_epoch=-1)
    amp_scaler = None
    if use_amp:
        amp_scaler = NativeScaler()

    train_cnt = 0
    for ep in range(epochs):
        model.train()
        train_acc_cnt = 0
        train_num = 0
        writer.add_scalar('Learning Rate', opt_sch.get_last_lr()[-1], ep)
        for i, (px, py) in enumerate(train_set):
            px:torch.Tensor = px.cuda()
            py:torch.Tensor = py.cuda()
            with amp.autocast():
                pred = model(px)
                loss = loss_func(pred, py)
            train_acc_cnt += accCounter(pred, py)
            train_num += len(pred)
            opt.zero_grad()
            if not amp_scaler is None:
                amp_scaler(loss, opt, clip_grad=None, parameters = model_parameters(model), create_graph = False)
            else:
                loss.backward()
                opt.step()
            if train_cnt % eval_time == 1:
                train_acc = train_acc_cnt / train_num
                print("Traning Epoch: %4d / %4d\t Batch %4d / %4d\t train loss: %.4f\t acc: %.4f\t lr: %f"%(
                        ep, epochs, i, batch_num, loss.item(), train_acc, opt_sch.get_last_lr()[-1]
                ))
                writer.add_scalar('Loss/Train Loss', loss, train_cnt)
                writer.add_scalar('Acc/Train Set Accuracy', train_acc, train_cnt)
                train_num = 0
                train_acc_cnt = 0
            train_cnt += 1
        model.eval()
        with torch.no_grad():
            ## +++++++++++ Load from Test set ++++++++=
            test_acc_cnt = 0
            test_loss:torch.Tensor = torch.zeros(1, device = device)
            eval_out_cnt = 0
            for j, (ptx, pty) in enumerate(test_set):
                ptx:torch.Tensor = ptx.cuda()
                pty:torch.Tensor = pty.cuda()
                pred = model(ptx)
                test_acc_cnt += accCounter(pred, pty)
                test_loss += loss_func(pred, pty)
                if (j + 1) % 20 == 0:
                    test_acc = test_acc_cnt / 2000
                    test_loss /= 2000
                    writer.add_scalar('Loss/Test loss', test_loss, eval_div * ep + eval_out_cnt)
                    writer.add_scalar('Acc/Test Set Accuracy', test_acc, eval_div * ep + eval_out_cnt)
                    print("Evalutaion in epoch: %4d / %4d\t evalutaion step: %d\t test loss: %.4f\t test acc: %.4f\t lr: %f"%(
                        ep, epochs, eval_out_cnt, test_loss.item(), test_acc, opt_sch.get_last_lr()[-1]
                    ))
                    eval_out_cnt += 1
                    test_acc_cnt = 0
                    test_loss.zero_()
        model.train()
        torch.save({
            'model': model.state_dict(),
            'optimizer': opt.state_dict()},
            "%schkpt_%d.pt"%(default_chkpt_path, train_cnt)
        )
        opt_sch.step()
        if augment_cancelled == False and ep > args.no_aug_epoch:
            augment_cancelled = True
            train_set = train_set_no_aug
    torch.save({
        'model': model.state_dict(),
        'optimizer': opt.state_dict()},
        "%smodel_4.pth"%(default_model_path)
    )
    writer.close()
    print("Output completed.")

if __name__ == "__main__":
    main()
