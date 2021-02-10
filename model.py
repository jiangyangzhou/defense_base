from models import WideResNet, ResNet18, ResNet34
import torch

def get_model_for_attack(model_name, pretrained=False):
    if model_name=='model1':
        model = ResNet34()
        model.load_state_dict(torch.load("models/weight/resnet34.pt"))
    elif model_name=='model2':
        model = ResNet18()
        model.load_state_dict(torch.load('models/weights/resnet18_AT.pt'))
    elif model_name=='model3':
        model = SmallResNet()
        model.load_state_dict(torch.load('models/weight/res_small.pth'))
    elif model_name=='model4':
        model = WideResNet()
        model.load_state_dict(torch.load('models/weight/trades_wide_resnet.pt'))
    elif model_name=='model5':
        model = WideResNet()
        model.load_state_dict(torch.load('models/weight/model-wideres-pgdHE-wide10.pt'))
    elif model_name=='model6':
        model = WideResNet(depth=28)
        model.load_state_dict(torch.load('models/weights/RST-AWP_cifar10_linf_wrn28-10.pt'))
    return model

if __name__=='__main__':
    model = model_dicts['Wu2020Adversarial_extra']
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)