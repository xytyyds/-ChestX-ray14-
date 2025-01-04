
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


def data_generate(reader):
    images = []
    labels = []

    start = True
    for row in reader:
        if start:
            start = False
            continue
        images.append("./data_pic/" + row[0])
        labels.append([int(row[i + 1]) for i in range(14)])
    
    #数据按照训练集：验证集：测试集 = 8 : 1 : 1来划分
    images = np.array(images)
    labels = np.array(labels)
    total_num = len(images)
    train_num = int(total_num * 0.8)
    val_num = int(total_num * 0.1)
    test_num = total_num - train_num - val_num

    train_images = images[:train_num]
    train_labels = labels[:train_num]
    val_images = images[train_num:train_num + val_num]
    val_labels = labels[train_num:train_num + val_num]
    test_images = images[train_num + val_num:]
    test_labels = labels[train_num + val_num:]

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


class CheXDataset(Dataset):
    def __init__(self, images, labels):
        super(CheXDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                                         torchvision.transforms.RandomHorizontalFlip(),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225])
                                                         ])  #数据归一化处理

    def __len__(self):
        return len(self.images)

    # 使用水平翻转做数据增强
    def __getitem__(self, item):
        image = Image.open(self.images[item]).convert('RGB')
        image = self.transform(image)

        label = self.labels[item]
        label = np.array(label, dtype=np.float32)
        label = torch.Tensor(label)
        return image, label
