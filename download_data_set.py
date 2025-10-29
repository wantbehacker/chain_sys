# -*- coding: utf-8 -*-
# download_cifar10.py
import os

from torchvision import datasets, transforms


def download_cifar10(data_root=r"./data_set"):
    """
    下载 CIFAR-10 数据集

    参数:
        data_root: 数据保存路径
    """
    os.makedirs(data_root, exist_ok=True)

    # 数据变换：转为Tensor，并归一化到 [0,1] 或者标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 均值
                             (0.2023, 0.1994, 0.2010))  # CIFAR-10 标准差
    ])

    # 下载训练集
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    print(f"训练集下载完成，样本数量: {len(train_dataset)}")

    # 下载测试集
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )
    print(f"测试集下载完成，样本数量: {len(test_dataset)}")

    return train_dataset, test_dataset


if __name__ == "__main__":
    download_cifar10()
