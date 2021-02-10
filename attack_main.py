import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from utils import prepare_cifar
from pgd_attack import pgd_attack
from models import  WideResNet, WideResNet34, WideResNet28
from model import get_model_for_attack
from tqdm import tqdm, trange
from eval_model import eval_model, eval_model_pgd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--step_size', type=int, default=0.003,
                    help='step size for pgd attack(default:0.03)')
    parser.add_argument('--epsilon', type=float, default=8/255.0,
                    help='max distance for pgd attack (default epsilon=8/255)')
    parser.add_argument('--perturb_steps', type=int, default=20,
                    help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--model_path', type=str, default="./models/weights/model-wideres-pgdHE-wide10.pt")
    parser.add_argument('--gpu_id', type=str, default="0,1")
    return parser.parse_args()

def get_test_data():
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    return test_loader

if __name__=='__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpu_num = max(len(args.gpu_id.split(',')), 1)
    device = torch.device('cuda')
    if args.model_name!="":
        model = get_model_for_attack(args.model_name).to(device)
        model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
    else:
        model =  WideResNet().to(device)  # Change to your model here
        model.load_state_dict(torch.load('models/weights/trades_wide_resnet.pt'))
        model = nn.DataParallel(model, device_ids=[0,1])
        #model.load_state_dict(torch.load('models/weights/RST-AWP_cifar10_linf_wrn28-10.pt')['state_dict'], strict=True)

        #model.load_state_dict(torch.load(args.model_path)['state_dict'], strict=False)   #Specify how to load your model here

    model.eval()
    test_loader = get_test_data()
    natural_acc, robust_acc, distance = eval_model_pgd(model, test_loader, device, args.step_size, args.epsilon, args.perturb_steps)
    print(f"Natural Acc: {natural_acc}, Robust acc: {robust_acc:.5f}, distance:{distance:.5f}")
