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

| model                | epoch | cutout | mixup | autoaugment | random-erase | train acc (%)   | val acc (%) | test acc (%)     | weitht:M |
| -------------------- | ----- | ------ | ----- | ----------- | ------------ | --------------- | ----------- | ---------------- | -------- |
| resnet34             | 80    |        |       |             |              |                 |             | 96.42            |          |
| resnet50             | 200   |        |       |             |              |                 |             | **96.49**        |          |

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

### 3.1 原pytorch-ricap的结果

| Model                           |    Error rate     |   Loss    | Error rate (paper) |
| :------------------------------ | :---------------: | :-------: | :----------------: |
| WideResNet28-10 baseline        |   3.82（96.18）   |   0.158   |        3.89        |
| WideResNet28-10 +RICAP          | **2.82（97.18）** |   0.141   |      **2.85**      |
| WideResNet28-10 +Random Erasing |   3.18（96.82）   | **0.114** |        4.65        |
| WideResNet28-10 +Mixup          |   3.02（96.98）   |   0.158   |        3.02        |

**修改网络的卷积层深度，并进行训练，可以得到以下结论：**

结论：lenet这种卷积量比较少，只有两层的，cpu利用率高，gpu利用率低。在这个基础上增加深度，用vgg那种直筒方式增加深度，发现深度越深，cpu利用率越低，gpu利用率越高。

**修改训练过程的batch size，可以得到以下结论：**

结论：bs会影响收敛效果。

### 3.5 StepLR优化下测试cutout和mixup

| architecture         | epoch | cutout | mixup | C10 test acc (%) |
| -------------------- | ----- | ------ | ----- | ---------------- |
| shake_resnet26_2x64d | 200   |        |       | 96.33            |
| shake_resnet26_2x64d | 200   | √      |       | 96.99            |
| shake_resnet26_2x64d | 200   |        | √     | 96.60            |
| shake_resnet26_2x64d | 200   | √      | √     | 96.46            |


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

复现：((**v100:gpu1**)  4min*300/60=20h) top1: **97.59%** 本项目目前最高值。


