import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from data_utils import prepare_cifar
from pgd_attack import pgd_attack
from models import WideResNet
from tqdm import tqdm, trange
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--step_size', type=int, default=0.003,
                    help='step size for pgd attack(default:0.03)')
    parser.add_argument('--epsilon', type=float, default=0.0314,
                    help='max distance for pgd attack (default epsilon=8/255)')
    parser.add_argument('--perturb_steps', type=int, default=20,
                    help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--model_path', type=str, default="./models/model-wideres-pgdHE-wide10.pt")
    parser.add_argument('--gpu_id', type=str, default="0,1")
    return parser.parse_args()

def eval_model_pgd(model, args, test_loader, device):
    correct_adv, correct = [], []
    distance = []
    num = 0
    with trange(10000) as pbar:
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            batch, c, h, w = x.shape
            x_adv = pgd_attack(model, x, label, args.step_size, args.epsilon, args.perturb_steps)
            model.eval()
            with torch.no_grad():
                output = model(x)
                output_adv = model(x_adv)
            distance.append(torch.max((x-x_adv).reshape(batch, -1).abs(), dim=1)[0])
            pred = output.argmax(dim=1)
            pred_adv = output_adv.argmax(dim=1)
            correct.append(pred == label)
            correct_adv.append(pred_adv == label)
            num += x.shape[0]
            pbar.set_description(f"Acc: {torch.cat(correct).float().mean():.5f}, Robust Acc:{torch.cat(correct_adv).float().mean():.5f}")
            pbar.update(x.shape[0])
    natural_acc = torch.cat(correct).float().mean()
    robust_acc = torch.cat(correct_adv).float().mean()
    distance = torch.cat(distance).max()
    return natural_acc, robust_acc, distance

if __name__=='__main__':
    args = parse_args()
    device = torch.device('cuda')
    model =  WideResNet().to(device)  # Change to your model here
    model = nn.DataParallel(model, device_ids=[0,1])
    model.load_state_dict(torch.load(args.model_path)['state_dict'], strict=False)   #Specify how to load your model here

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    natural_acc, robust_acc, distance = eval_model_pgd(model, args, test_loader, device)
    print(f"Natural Acc: {natural_acc}, Robust acc: {robust_acc:.5f}, distance:{distance:.5f}")
