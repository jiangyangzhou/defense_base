# Attack and Defense 
Attack and Defense API for adversarial training project for Machine Learning in Action course in SJTU 


### Set up:
** Requirement ** `python3`, `pytorch > 1.2`, `tqdm`  
1. Install python  
Recomand install anaconda, see https://www.anaconda.com/products/individual#Downloads
2. Create new environment and install pytorch, tqdm:
```
conda create -n t17 python=3.8
conda install pytorch=1.7 torchvision cudatoolkit=10.2 -c pytorch
pip install tqdm
```
### Notes
1. About args  
代码中使用的是Python包`argparse`，来在命令行解析参数。例如设置attack的model: `python attack_main.py --model_name=model1`  
2. About gpu  
推荐使用GPU训练，多卡训练，需设置gpu_id, eg: `python pgd_train.py --gpu_id=0,1`.
在这里使用 Pytorch 的nn.DataParallel 实现多卡并行， 涉及代码如下。nn.DataParallel 实际上 wrap the pytorch model as `model.module`.
```
device = torch.device('cuda')
model = ResNet18().to(device)    
model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
```
3. About log  
关于日志保存，请参见代码中`Logger` class, 也可自行实现日志部分  
4. 其他问题我会更新在这里  

#### ML Course
See [ML2021.md](https://github.com/jiangyangzhou/defense_base/blob/main/ML2021.md)


