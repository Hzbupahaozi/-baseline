import os
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchsummary import summary
from vgg_model import Vgg16_Net
from resnet_model import resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from ResNeXt_model import resnext18, resnext34, resnext50, resnext101, resnext152
from wideresnet_model import wide_resnet28_10, wide_resnet40_10, wide_resnet28_20, wide_resnet40_14
# from efficientnet_model import efficientnet_b0
from efficientnet_pytorch import EfficientNet
from torchvision.models import densenet121
from resnet_cbam_model import CBAM_ResNet50

import re
from utils.cutout import Cutout
from utils.mixup import mixup_data, mixup_criterion
from torch.autograd import Variable
import time


def main():
    batch_size = 64
    epochs = 20
    learning_rate = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),  # 随机裁剪
                                     # transforms.RandomGrayscale(),
                                     transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                     transforms.ToTensor(),  # totensor格式
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]),
        # 因为不是迁移学习也不是与训练模型，所以就是可以不用减去RGB的均值
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((224, 224)),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # train_data = datasets.CIFAR10(root='./data/', train=True, transform=data_transform["train"], download=False)
    # test_data = datasets.CIFAR10(root='./data/', train=False, transform=data_transform["val"], download=False)
    image_path = './data/'
    train_data = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                      transform=data_transform['train'])
    test_data = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                     transform=data_transform['val'])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_data, batch_size=5, shuffle=False, num_workers=0)


    # model = Vgg16_Net()
    # --------------------------------------------
    # weights_path = "pre_train/efficientnet-b0.pth"
    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10)
    # model_name = 'efficientnet-b0'

    # weights_dict = torch.load(weights_path, map_location="cuda")
    # # model.load_state_dict(weights_dict)
    # load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
    # print(model.load_state_dict(load_weights_dict, strict=False))
    # in_channel = model.classifier.in_features
    # model.classifier = nn.Linear(in_channel, 10)
    # --------------------------------------------
    # model = densenet121()
    # model_name = 'densenet121'
    # pthfile = "pre_train/densenet121.pth"
    # pattern = re.compile(
    #     r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    # state_dict = torch.load(pthfile)
    # for key in list(state_dict.keys()):
    #     res = pattern.match(key)
    #     if res:
    #         new_key = res.group(1) + res.group(2)
    #         state_dict[new_key] = state_dict[key]
    #         del state_dict[key]
    # model.load_state_dict(state_dict)
    # in_channel = model.classifier.in_features
    # model.classifier = nn.Linear(in_channel, 10)
    # ------------------------------------------------

    # --------------------------------------
    # model = CBAM_ResNet50()
    # model_name = 'CBAM_ResNet50_try'
    # # pre_train_weights = "checkpoint/CBAM_ResNet50_20_lr/model.pth"
    # # model.load_state_dict(torch.load(pre_train_weights, map_location='cpu'))
    # # 因为要添加Imagenet的预训练权重，所以要修改最后一层
    # in_channel = model.linear.in_features
    # model.linear = nn.Linear(in_channel, 9)
    # ------------------------------------------

    # model = resnet50()/resnext50()
    model = resnet50()
    pre_train_weights = "./pre_train/resnet50.pth"
    model.load_state_dict(torch.load(pre_train_weights, map_location='cpu'))
    model_name = 'resnet50_apple2'
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 9)
    # -------------------------------------------

    # --------------------------------------------
    # model = wide_resnet28_10()
    # model_name = 'wide_resnet28_10'
    # --------------------------------------------

    # tensorboard画出模型
    # dummy_input = torch.rand(1, 3, 32, 32)

    # apple
    dummy_input = torch.rand(1, 3, 224, 224)
    with SummaryWriter(comment=model_name) as w:
        w.add_graph(model, dummy_input)

    model.to(device)
    # print(model, '\n')
    summary(model, input_size=(3, 224, 224), device='cuda')
    writer = SummaryWriter(log_dir='./tensorboard_event', filename_suffix=str(epochs))

    # 定义损失函数，分类问题使用交叉信息熵，回归问题使用MSE
    criterion = nn.CrossEntropyLoss()
    # torch.optim来做算法优化,该函数甚至可以指定每一层的学习率，这里选用Adam来做优化器，还可以选其他的优化器
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True,
    #                                            threshold=0.00005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    # 设置每50个epoch调整学习率，lr=0.1*lr
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15,
    #                                       gamma=0.1)  # 设置学习率下降策略,就是没过step_size个epoch就调整lr为lr*gamma
    num = 0
    iter = 0
    best_acc = 0
    for epoch in range(epochs):

        # if epoch == 99:
        #     learning_rate = 0.0002
        #     optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        print(f"学习率：{optimizer.param_groups[0]['lr']}")
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        test_loss = 0
        test_correct = 0
        test_total = 0
        with tqdm(train_loader) as tepoch:
            for i, (inputs, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch{epoch}")
                num += 1
                inputs, labels = inputs.to(device), labels.to(device)

                # # 前向传播
                outputs = model(inputs)
                # # 计算损失函数
                loss = criterion(outputs, labels)
                # 清空上一轮的梯度
                optimizer.zero_grad()
                # 反向传播
                loss.backward()
                # ---------------------------------------------------
                writer.add_scalar('{}/train_loss'.format(model_name), loss.item(), num)
                # 参数更新
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                # progress_bar(i, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (train_loss / (i + 1), 100. * train_correct / train_total, train_correct, train_total))
        print('epoch{}:'.format(epoch), 'train loss: %.3f | train acc: %.3f%% (%d/%d)'
              % (train_loss / (i + 1), 100. * train_correct / train_total, train_correct, train_total))

        writer.add_scalar('{}/train_accuracy'.format(model_name), (100. * train_correct / train_total), epoch)

        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                iter += 1
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss2 = criterion(outputs, labels)
                writer.add_scalar('{}/test_loss'.format(model_name), loss2.item(), num)
                test_loss += loss2.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                # progress_bar(i, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss / (i + 1), 100. * test_correct / test_total, test_correct, test_total))
            print('epoch{}:'.format(epoch), 'test Loss: %.3f | test cc: %.3f%% (%d/%d)'
                  % (test_loss / (i + 1), 100. * test_correct / test_total, test_correct, test_total))

        acc = 100. * test_correct / test_total
        if acc >= best_acc:
            best_acc = acc
            print('-----Saving model-----')
            if not os.path.isdir('checkpoint/{}'.format(model_name)):
                os.mkdir('checkpoint/{}'.format(model_name))
            torch.save(model, './checkpoint/{}/model.pth'.format(model_name))
        writer.add_scalar('{}/test_accuracy'.format(model_name), best_acc, epoch)
        scheduler.step()
    print("finished best_acc:{}".format(best_acc))


# 时间
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


if __name__ == '__main__':
    start_time = time.time()
    main()
    time_consumed = time.time() - start_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(time_consumed)))
