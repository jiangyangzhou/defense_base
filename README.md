# defense_base
Attack and Defense API for adversarial training project for Machine Learning in Action course in SJTU 

### Set up:
1. Install python
Recomand install anaconda, see https://www.anaconda.com/products/individual#Downloads
2. Create new environment and install pytorch:
```
conda create -n myenv python=3.8

```


## Attack Task
You need design attack algorithm to attack provided 6 models. Note that we use constraint of `l-inf` distance `< 8./225`. 

### Dataset: CIFAR10
Use `prepare_cifar` in utils.py to get train_loader and test_loader of CIFAR10.

```
from utils import prepare_cifar
train_loader, test_loader = prepare_cifar(batch_size = 128, test_batch_size = 256)
```

### Defense models
1. model1:  vanilla resnet34
2. model2:  PGD adversarial trained resnet18
3. model3:  Unkonown resnet
4. model4:  TRADES [TRADES:https://arxiv.org/abs/1901.08573] 
5. model5:  PGD_HE [PGD_HE:https://arxiv.org/abs/2002.08619]
6. model6:  RST_AWP [RST_AWP:https://arxiv.org/abs/2004.05884]

Step1, Download model weights here [Jbox:https://jbox.sjtu.edu.cn/l/PFOOnZ].  
Step2, 

#### Attack Baseline
See pgd_attack.py and attack_main.py for PGD attack baseline code



### How to run PGD attack to test robustness of your model
1. Open attack_main.py, specify how to load your model.
  A example in attack_main.py:
```
model =  WideResNet().to(device)  # Change to your model here
model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
model.load_state_dict(torch.load(args.model_path)['state_dict'], strict=False)   #Specify how to load your model here
```
2. Run attack_main.py:
```
python attack_main.py --model_path=your_weight_path --gpu_id=0      #For multiple gpus, set --gpu_id=1,2,3
```
It will give natural acc and robust acc for your model.

### How to train a robust model 
Remain for you. 
