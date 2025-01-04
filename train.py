import numpy as np
import torch
import csv
from dataset import CheXDataset, data_generate
from torch.utils import data
from net.model import DenseNet121, DenseNet121_torch_version
from tqdm import tqdm
from pytorchtools import EarlyStopping
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt  # 新增，用于绘制图表

# 获取优化器当前的学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
    # 根据是否有可用的GPU来选择设备，优先使用GPU（cuda:0），如果没有则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 打开训练数据的CSV文件，用于读取数据
    readFile = open("train.csv", "r", newline='')
    reader = csv.reader(readFile)

    # 调用data_generate函数生成训练集和验证集的图像、标签数据
    train_images, train_labels, val_images, val_labels = data_generate(reader)
    num_train = len(train_labels)
    num_val = len(val_labels)
    train_batch_size = 32

    # 创建训练集的数据集对象，传入训练集图像和标签数据，用于后续的数据加载和模型训练
    trainDataset = CheXDataset(train_images, train_labels)
    # 创建验证集的数据集对象，传入验证集图像和标签数据，用于后续的数据加载和模型验证
    valDataset = CheXDataset(val_images, val_labels)
    # 创建训练集的数据加载器，设置批量大小为train_batch_size，并打乱数据顺序，方便模型训练时以小批次的方式获取数据
    trainData_iter = data.DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True)
    # 创建验证集的数据加载器，这里批量大小设为1，可根据实际情况调整，同样打乱数据顺序（如果需要）
    valData_iter = data.DataLoader(valDataset, batch_size=1, shuffle=True)

    # 定义损失函数为二元交叉熵损失（BCELoss），并设置size_average为True（在较新版本的PyTorch中已弃用，可根据实际需求调整）
    model_loss = torch.nn.BCELoss(size_average=True)
    # 实例化DenseNet121_torch_version模型，传入类别数量14（根据具体任务类别数确定）
    model = DenseNet121_torch_version(14)
    # 将模型移动到指定的设备（GPU或CPU）上，以便后续在该设备上进行计算
    model.to(device)

    epoches = 40  # 定义训练的总轮数
    epoch_step_train = num_train // train_batch_size  # 计算每一轮训练中，完整遍历训练集需要的迭代次数
    epoch_step_val = num_val // train_batch_size  # 计算每一轮验证中，完整遍历验证集需要的迭代次数

    # 以下是超参数调优相关的部分，这里使用ReduceLROnPlateau学习率调整策略
    # 定义优化器为Adam，设置学习率、beta参数、eps以及权重衰减等超参数，可根据实际情况调整这些参数来优化模型训练效果
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # 使用ReduceLROnPlateau学习率调整策略，当验证集损失在一定轮数（patience=5）内没有下降时，学习率乘以factor（0.1）进行降低
    lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')

    train_losses = []  # 用于记录每一轮训练的平均损失，方便后续绘制曲线
    val_losses = []  # 用于记录每一轮验证的平均损失，方便后续绘制曲线

    for epoch in range(epoches):
        train_loss = 0
        val_loss = 0
        print('Start Training')
        model.train()
        # 如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加model.train()。
        # model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
        with tqdm(total=epoch_step_train, desc=f'Epoch {epoch + 1}/{epoches}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(trainData_iter):  # batch:(X, y)
                if iteration > epoch_step_train:
                    break

                images, labels = batch

                outputs = model(images.to(device))

                loss = model_loss(outputs, labels.to(device))
                #
                # loss = loss / len(outputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
                pbar.set_postfix(**{'loss': train_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
        # 计算本轮训练的平均损失并添加到列表中
        avg_train_loss = train_loss / epoch_step_train
        train_losses.append(avg_train_loss.item())

        print('Finished Training')

        print('Start Validation')
        model.eval()
        # 如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。
        # model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。
        # 对于Dropout，model.eval() 是利用到了所有网络连接，即不进行随机舍弃神经元。
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{epoches}', postfix=dict, mininterval=0.3) as pbar:
            val_loss = 0
            for iteration, batch in enumerate(valData_iter):  # batch:(X, y)
                if iteration > epoch_step_val:
                    break

                images, labels = batch

                outputs = model(images.to(device))
                loss = model_loss(outputs, labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                val_loss += loss
                pbar.set_postfix(**{'loss': val_loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

        # 计算本轮验证的平均损失并添加到列表中
        avg_val_loss = val_loss / epoch_step_val
        val_losses.append(avg_val_loss.item())

        print('Finished Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(epoches))
        print('Train Loss: %.3f || Val Loss: %.3f ' % (avg_train_loss, avg_val_loss))

        torch.save(model, 'weights/CheXNet_Val_Loss:_%.3f.pth' % (avg_val_loss))

        lr_scheduler.step(avg_val_loss)
        # early_stopping(val_loss / epoch_step_val, model)
        # if early_stopping.early_stop:
        #     break

    # 保存最终模型
    torch.save(model, 'weights/CheXNet.pth')

    # 绘制训练集和验证集的损失曲线
    plt.plot(range(1, epoches + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epoches + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.show()
