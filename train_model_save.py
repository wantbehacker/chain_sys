# -*- coding: utf-8 -*-


import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model


# ===================== 工具函数 =====================
def load_config(cfg_path="./config.json"):
    """读取训练配置"""
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataloaders(image_size, ds_name, data_root, batch_size, num_workers=0):
    """加载数据集"""
    data_root = Path(data_root)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if ds_name in ["mnist", "emnist"] else
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if ds_name == "mnist":
        train_set = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(data_root, train=False, download=True, transform=transform)
        num_classes = 10
    elif ds_name == "emnist":
        train_set = datasets.EMNIST(data_root, split="byclass", train=True, download=True, transform=transform)
        test_set = datasets.EMNIST(data_root, split="byclass", train=False, download=True, transform=transform)
        num_classes = 62
    elif ds_name == "cifar10":
        train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(data_root, train=False, download=True, transform=transform)
        num_classes = 10
    elif ds_name == "cifar100":
        train_set = datasets.CIFAR100(data_root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(data_root, train=False, download=True, transform=transform)
        num_classes = 100
    else:
        train_set = datasets.ImageFolder(data_root / "train", transform=transform)
        test_set = datasets.ImageFolder(data_root / "val", transform=transform)
        num_classes = len(train_set.classes)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes


# ===================== 核心函数 =====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            total += target.size(0)
            correct += (preds == target).sum().item()
    return 100.0 * correct / total


# ===================== 主训练函数 =====================
def train(cfg_path="./config.json"):
    """主训练入口"""
    cfg = load_config(cfg_path)

    dataset_name = cfg["model_name"].split("-")[0]
    data_root = cfg["data_root"]
    batch_size = cfg["batch_size"]
    image_size = cfg["image_size"]
    total_epochs = cfg["total_epochs"]
    lr = cfg["learning_rate"]
    momentum = cfg["momentum"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前训练设备: {device}")

    # ===== 加载数据集 =====
    train_loader, test_loader, num_classes = get_dataloaders(image_size, dataset_name, data_root, batch_size)
    print(f"数据集: {dataset_name}, 类别数: {num_classes}")

    # ===== 构建模型 =====
    model = build_model(cfg["model_name"], num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # ===== 开始训练 =====
    print(f"开始训练，共 {total_epochs} 轮。")
    for epoch in range(1, total_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{total_epochs} | Loss={loss:.4f} | Acc={acc:.2f}%")

    os.makedirs("./save_model", exist_ok=True)
    save_path = f"./save_model/{cfg['model_name']}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")


# ===================== 直接运行 =====================
if __name__ == "__main__":
    train()
