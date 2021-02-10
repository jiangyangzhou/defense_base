# Attack and Defense （Preview）
Attack and Defense API for adversarial training project for Machine Learning in Action course in SJTU 


### Set up:
1. Install python
Recomand install anaconda, see https://www.anaconda.com/products/individual#Downloads
2. Create new environment and install pytorch, tqdm:
```sh
conda create -n myenv python=3.8
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install tqdm
```

# Attack Task
In this task, you need design attack algorithm to attack provided 6 models.  
Note that we use constraint of `l-inf` distance `< 8./225`. 

### Dataset: CIFAR10
Use `prepare_cifar` in utils.py to get `train_loader` and `test_loader` of CIFAR10.

```python
from utils import prepare_cifar
train_loader, test_loader = prepare_cifar(batch_size = 128, test_batch_size = 256)
```

### Defense models
1. model1:  vanilla resnet34
2. model2:  PGD adversarial trained resnet18
3. model3:  Unkonown resnet
4. model4:  [TRADES](https://arxiv.org/abs/1901.08573)
5. model5:  [PGD_HE](https://arxiv.org/abs/2002.08619)
6. model6:  [RST_AWP](https://arxiv.org/abs/2004.05884)

Step1, download model weights here [Jbox](https://jbox.sjtu.edu.cn/l/PFOOnZ)  
Step2, move model weights to path `models/weights/`  
Run model: see `get_model_for_attack` in model.py  

#### Attack Baseline
See pgd_attack.py and attack_main.py for PGD attack baseline code

#### Test your attack
See attack_main.py, and replace pgd_attack method to your own attack method.  
And run attack_main.py to test your attack, set model_name to [model1, model2, model3, model4, model5, model6]. Like:
```sh
python attack_main.py --model_name=model1
```



# Defense Task
In this task, you need to train a robust model under l-inf attack(8/255) on CIFAR10.

### Evaluate your model
We'll use various attack methods to evaluate the robustness of your model.  
Include PGD attack, and others.

#### How to run PGD attack to test robustness of your model
1. Open attack_main.py, specify how to load your model.
  A example in attack_main.py:
```python
model =  WideResNet().to(device)  # Change to your model here
model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
model.load_state_dict(torch.load(args.model_path)['state_dict'], strict=False)   # Specify how to load your model here
```
2. Run attack_main.py:
```python
python attack_main.py --model_path=your_weight_path --gpu_id=0      #For multiple gpus, set --gpu_id=1,2,3
```
It will give natural acc and robust acc for your model.

### How to train a robust model 
See pgd_train.py 
