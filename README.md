# 科大讯飞算法赛baseline
# 简单记录一下过程
## 0. Requirements

- Python 3.6+
- torch=1.8.0+cu111
- torchvision+0.9.0+cu111
- tqdm=4.26.0
- PyYAML=6.0
- einops
- torchsummary


## 1. Implements

### 1.0 Models
### 测试多种数据增强
- train_aug1:
  - Resize((224,224))
  - RandomHorizontalFlip
  - Normalization((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  
- val_aug1:
  - Resize((224,224))
  - Normalization((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

- train_aug2:
  - RandomCrop(crop_size=224)
  - RandomHorizontalFlip()
  - RandomDistort()
  - RandomBlur(prob=0.1)
  - transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], min_val=[0, 0, 0], max_val=[255.0, 255.0, 255.0])

- val_aug2:
  - Resize((256,256))
  - CenterCrop(crop_size=224)
  - transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], min_val=[0, 0, 0], max_val=[255.0, 255.0, 255.0])


| model                | epoch | cutout | mixup | autoaugment | train acc (%)   | val acc (%) | test acc (%)     | weitht:M |
| -------------------- | ----- | ------ | ----- | ----------- | --------------- | ----------- | ---------------- | -------- |
| resnet34             | 80    |        |       | aug1        |                 |             | 96.42            |          |
| resnet50             | 60    |        |       |             |                 |             | **96.49**        |          |
| resnet50             | 60    |        |       | aug1        | 99.674          | 96.653      |                  |          |


### todolist
首先是余弦退火的参数要改，然后就是加上cutout，然后就是normalization的参数，然后就是加了几个数据增强的还没用
warmup？


### 1.1 Tricks

- Warmup
- Cosine LR Decay
- SAM
- Label Smooth
- KD
- Adabound
- Xavier Kaiming init
- lr finder

### 1.2 Augmentation

- Auto Augmentation
- Cutout
- Mixup
- RICAP
- Random Erase
- ShakeDrop

## 3. Results

**修改网络的卷积层深度，并进行训练，可以得到以下结论：**

结论：lenet这种卷积量比较少，只有两层的，cpu利用率高，gpu利用率低。在这个基础上增加深度，用vgg那种直筒方式增加深度，发现深度越深，cpu利用率越低，gpu利用率越高。

**修改训练过程的batch size，可以得到以下结论：**

结论：bs会影响收敛效果。

- lr:
  - warmup (20 epoch)
  - cosine lr decay
  - lr=0.1
  - total epoch(300 epoch)
- bs=128
- aug:
  - Random Crop and resize
  - Random left-right flipping
  - AutoAugment
  - Normalization
  - Random Erasing
  - Mixup
- weight decay=5e-4 (bias and bn undecayed)
- kaiming weight init
- optimizer: nesterov



