import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from pgd_attack import pgd_attack
from models import ResNet18
from tqdm import tqdm, trange

from attack_main import eval_model_pgd
from loss import L2Loss, WeightConstrainLoss, PGDLoss
from data_utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')               
    parser.add_argument('--step_size', type=int, default=0.007,
                    help='step size for pgd attack(default:0.03)')
    parser.add_argument('--perturb_steps', type=int, default=10,
                    help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--epsilon', type=float, default=8./255.,
                    help='max distance for pgd attack (default epsilon=8/255)')
    parser.add_argument('--lr', type=float, default=0.1,
                    help='iterations for pgd attack (default pgd20)')
    #parser.add_argument('--lr_steps', type=str, default=,
    #                help='iterations for pgd attack (default pgd20)')    
    parser.add_argument('--epoch', type=int, default=100,
                    help='epochs for pgd training ')  
    parser.add_argument('--epoch', type=int, default=100,
                    help='epochs for pgd training ')  
    parser.add_argument('--momentum', type=float, default=0.9,
                    help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay ratio')     
    parser.add_argument('--delta', type=float, default=0.1,
                    help='delta')              
    #parser.add_argument('--model_path', type=str, default="./models/model-wideres-pgdHE-wide10.pt")
    parser.add_argument('--gpu_id', type=str, default="0,1")
    return parser.parse_args()



def train_adv_epoch(model, args, train_loader, device, optimizer, epoch):
    model.train()
    corrects_adv, corrects = 0, 0
    data_num = 0 
    loss_sum = 0
    with trange( len(train_loader.dataset)) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            x, y = data.to(device), target.to(device)
            data_num += x.shape[0]
            optimizer.zero_grad()
            x_adv = pgd_attack(model, x, y, args.step_size, args.epsilon, args.perturb_steps,
                    random_start=0.001, distance='l_inf')
            model.train()
            output_adv = model(x_adv)
            loss = nn.CrossEntropyLoss()(output_adv, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            with torch.no_grad():
                model.eval()
                output_adv = model(x_adv)
                pred_adv = output_adv.argmax(dim=1)
                pred = torch.argmax(model(x), dim=1)
                corrects_adv += (pred_adv==y).float().sum()
                corrects += (pred==y).float().sum()
            pbar.set_description(f"Train Epoch:{epoch}, Loss:{loss.item():.3f}, " + 
                            f"acc:{corrects / float(data_num):.4f}, ")+
                        #" r_acc:{corrects_adv / float(data_num):.4f}")
            pbar.update(x.shape[0])
    acc, adv_acc = corrects / float(data_num), corrects_adv / float(data_num)
    mean_loss = loss_sum / float(batch_idx+1)
    return acc, adv_acc, mean_loss


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




if __name__=="__main__":
    args = parse_args()
    log_dir = "logs/%s_resnet18"%time.strftime("%b%d-%H%M", time.localtime())

    check_mkdir(log_dir)
    log = Logger(log_dir+'/train.log')

    device = torch.device('cuda')
    model = ResNet18().to(device)
    model = nn.DataParallel(model, device_ids=[0,1])
    train_loader, test_loader = prepare_cifar(args.batch_size, args.test_batch_size)
    log.print(args)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    best_epoch, best_robust_acc = 0, 0.
    criterion_weight = WeightLoss(args.delta)
    criterion_l2 = L2Loss(args.delta)
    for e in range(args.epoch):
        adjust_learning_rate(optimizer, e)
        train_acc, train_robust_acc, loss = train_adv_epoch(model, args, train_loader, device,  optimizer, e)
        test_acc, test_robust_acc, _ = eval_model_pgd( model, args,  test_loader, device)
        if test_robust_acc > best_robust_acc:
            best_robust_acc = test_robust_acc
            best_epoch = e
        if e > 50:
            torch.save(model.module.state_dict(),
                       os.path.join(log_dir, f"resnet18-e{e}-{test_acc:.4f}_{test_robust_acc:.4f}-best.pt"))
        log.print(f"Epoch:{e}, loss:{loss:.5f}, train_acc:{train_acc:.4f}, train_rubust_acc:{train_robust_acc:.4f},  " + 
                f"test_acc:{test_acc:.4f}, test_rubust_acc:{test_robust_acc:.4f}, best_robust_acc:{best_robust_acc:.4f} in epoch {best_epoch}.")
    torch.save(model.module.state_dict(), f"{log_dir}/resnet18_e{args.epoch - 1}_{test_acc:.4f}_{test_robust_acc:.4f}-final.pt")
        
