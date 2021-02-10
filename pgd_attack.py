import torch
import torch.nn as nn
import torch.nn.functional as F

def pgd_attack(model, x, y, step_size, epsilon, perturb_steps,
                random_start=None, distance='l_inf'):
    model.eval()
    batch_size = len(x)
    if random_start:
        x_adv = x.detach() + random_start * torch.randn(x.shape).cuda().detach()
    else:
        x_adv = x.detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

