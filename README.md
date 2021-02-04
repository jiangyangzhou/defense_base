# defense_base
Defense API for adversarial training project for Machine Learning in Action course in SJTU 

### Dependents:
```
Python 3.8 
Pytorch 1.7
```

### How to run PGD attack to test your model robustness:
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
