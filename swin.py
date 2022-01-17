#-*-coding:utf-8-*-
"""
    This is a template for torch train.py
    This is a timm-bounded version
"""
import os
import torch
import argparse
from functools import partial

from torch import optim

from torch.cuda import amp
from timm.utils import NativeScaler
from timm.models import model_parameters
from py.train_utils import *
from swin.swinLayer import SwinTransformer

from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.data import create_loader, create_dataset

default_chkpt_path = "./check_points/swin_"
default_model_path = "./model/swin_"

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type = int, default = 20, help = "Training lasts for . epochs")
parser.add_argument("--batch_size", type = int, default = 100, help = "Batch size for range image packs.")
parser.add_argument("--eval_time", type = int, default = 50, help = "Eval every <eval_time> batches.")
parser.add_argument("--cooldown_epoch", type = int, default = 10, help = "Epochs to cool down lr after cyclic lr.")
parser.add_argument("--eval_div", type = int, default = 200, help = "Output every <...> times in evaluation. Match ticks with training")
parser.add_argument("--name", type = str, default = "model_1.pth", help = "Model name for loading")
parser.add_argument("--weight_decay", type = float, default = 3e-2, help = "Weight Decay in AdamW")
parser.add_argument("--max_lr", type = float, default = 55e-5, help = "Max learning rate")
parser.add_argument("--min_lr", type = float, default = 1e-5, help = "Min learning rate")
parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
parser.add_argument("-l", "--load", default = False, action = "store_true", help = "Load checkpoint or trained model.")
parser.add_argument("-s", "--use_scaler", default = False, action = "store_true", help = "Use AMP scaler to speed up")
args = parser.parse_args()

def get_sch_lr(sch:CosineLRScheduler, start_lr:float, i:int)->float:
    return sch.get_epoch_values(i)[0] * start_lr

def main():
    epochs              = args.epochs
    eval_time           = args.eval_time
    eval_div            = args.eval_div
    use_amp             = args.use_scaler
    batch_size          = args.batch_size
    del_dir             = args.del_dir
    use_load            = args.load
    load_path           = default_model_path + args.name

    # ======= load train set and test set =======
    train_set = create_dataset("Imagenette_train", "../dataset/imagenette2-320/", "train", is_training = True)
    test_set = create_dataset("Imagenette_val", "../dataset/imagenette2-320/", "val", is_training = False)
    train_loader = create_loader(train_set, 224, batch_size, True, 
        use_prefetcher = False, num_workers = 8, pin_memory = True, persistent_workers = True)
    test_loader = create_loader(test_set, 224, 50, True, use_prefetcher = False, 
        num_workers = 8, pin_memory = True, persistent_workers = True)

    device = None
    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(-1)
    device = torch.device(0)
    
    model = SwinTransformer(7, 96, 224, (2, 2, 4, 2)).cuda()
    if use_load == True and os.path.exists(load_path):
        model.loadFromFile(load_path)
    else:
        print("Not loading or load path '%s' does not exist."%(load_path))

    # ======= Loss function ==========
    loss_func = LabelSmoothingCrossEntropy().cuda()
    eval_loss_func = torch.nn.CrossEntropyLoss().cuda()

    # ======= Optimizer and scheduler ========
    opt = optim.AdamW(model.parameters(), lr = 5e-6, betas = (0.9, 0.999), weight_decay=args.weight_decay)
    # min_max_ratio = args.min_lr / args.max_lr
    # lec_sch_func = CosineLRScheduler(opt, t_initial = epochs // 2, t_mul = 1, lr_min = min_max_ratio, decay_rate = 0.1,
    #         warmup_lr_init = min_max_ratio, warmup_t = 5, cycle_limit = 2, t_in_epochs = True)
    # opt_sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda = partial(get_sch_lr, lec_sch_func, args.max_lr), last_epoch=-1)
    # epochs = lec_sch_func.get_cycle_length() + args.cooldown_epoch
    
    # ====== tensorboard summary writer ======
    writer = getSummaryWriter(epochs, del_dir)

    amp_scaler = None
    if use_amp:
        amp_scaler = NativeScaler()
    train_cnt = 0
    batch_num = len(train_loader)
    for ep in range(epochs):
        model.train()
        train_acc_cnt = 0
        train_num = 0
        # writer.add_scalar('Learning Rate', opt_sch.get_last_lr()[-1], ep)
        
        for i, (px, py) in enumerate(train_loader):
            px:torch.Tensor = px.cuda()
            py:torch.Tensor = py.cuda()
            with amp.autocast():
                pred = model(px)
                loss:torch.Tensor = loss_func(pred, py)
            train_acc_cnt += accCounter(pred, py)
            train_num += len(pred)

            opt.zero_grad()
            if not amp_scaler is None:
                amp_scaler(loss, opt, clip_grad=None, parameters = model_parameters(model), create_graph = False)
            else:
                loss.backward()
                opt.step()
            torch.cuda.synchronize()            # Maybe Prefetch Loader needs sync

            if train_cnt % eval_time == 1:
                train_acc = train_acc_cnt / train_num
                print("Traning Epoch: %4d / %4d\t Batch %4d / %4d\t train loss: %.4f\t acc: %.4f\t"%(
                        ep, epochs, i, batch_num, loss.item(), train_acc
                ))
                writer.add_scalar('Loss/Train Loss', loss, train_cnt)
                writer.add_scalar('Acc/Train Set Accuracy', train_acc, train_cnt)
                train_num = 0
                train_acc_cnt = 0
            train_cnt += 1

        model.eval()
        with torch.no_grad():
            test_acc_cnt = 0
            eval_out_cnt = 0
            test_cnt = 0
            test_loss:torch.Tensor = torch.zeros(1, device = device)
            for j, (ptx, pty) in enumerate(test_loader):
                ptx:torch.Tensor = ptx.cuda()
                pty:torch.Tensor = pty.cuda()

                pred = model(ptx)
                test_acc_cnt += accCounter(pred, pty)
                test_loss += eval_loss_func(pred, pty)
                test_cnt += len(pty)
                if (j + 1) % 5 == 0:
                    test_acc = test_acc_cnt / test_cnt
                    test_loss /= test_cnt

                    writer.add_scalar('Loss/Test loss', test_loss, eval_div * ep + eval_out_cnt)
                    writer.add_scalar('Acc/Test Set Accuracy', test_acc, eval_div * ep + eval_out_cnt)
                    print("Evalutaion in epoch: %4d / %4d\t evalutaion step: %d\t test loss: %.4f\t test acc: %.4f\t"%(
                        ep, epochs, eval_out_cnt, test_loss.item(), test_acc, 
                    ))
                    eval_out_cnt += 1
                    test_acc_cnt = 0
                    test_cnt = 0
                    test_loss.zero_()
        model.train()
        # torch.save({
        #     'model': model.state_dict(),
        #     'optimizer': opt.state_dict()},
        #     "%schkpt_%d.pt"%(default_chkpt_path, train_cnt)
        # )
    torch.save({
        'model': model.state_dict(),
        'optimizer': opt.state_dict()},
        "%smodel_2.pth"%(default_model_path)
    )
    writer.close()
    print("Output completed.")

if __name__ == "__main__":
    main()
