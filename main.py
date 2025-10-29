# -*- coding: utf-8 -*-
"""
通用分类训练脚本（交替执行版）
支持：mnist / CIFAR10 / CIFAR100 / ImageFolder
模型由 models.build_model() 提供
配置文件：config.json
"""
import base64
import copy
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from flask import Flask, jsonify
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from handle_chain import ChainClient
from models import build_model  # 从 models 模块导入

# =================== 全局缓存 ===================
return_data = {
    "global_acc": {"normal": [], "replace": []},
    "client_loss": {"normal": [], "replace": []},
    "now_epochs": "",
    "infer_results": {"normal": {}, "replace": {}},
    "status": {}
}

mid_data = {
    "global_acc": {"normal": [], "replace": []},
    "client_loss": {"normal": [], "replace": []},
    "now_epochs": "",
    "infer_results": {"normal": {}, "replace": {}},
    "status": {}
}

# =================== Flask API ===================
app_return = Flask("app_return")


@app_return.route("/", methods=["GET"])
def get_return_data():
    return jsonify(return_data)


# =================== 配置加载 ===================
def load_config():
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


# =================== 数据加载 ===================
def get_dataloaders(image_size, ds_name, data_root, batch_size, num_workers=0):
    data_root = Path(data_root)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if ds_name in ["mnist", "emnist"] else
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if ds_name == "mnist":
        train_dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_root, train=False, download=True, transform=transform)
        class_nums = 10
    elif ds_name == "cifar10":
        train_dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_root, train=False, download=True, transform=transform)
        class_nums = 10
    elif ds_name == "cifar100":
        train_dataset = datasets.CIFAR100(data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_root, train=False, download=True, transform=transform)
        class_nums = 100
    elif ds_name == "emnist":
        train_dataset = datasets.EMNIST(data_root, split="byclass", train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST(data_root, split="byclass", train=False, download=True, transform=transform)
        class_nums = 62
    else:
        train_dataset = datasets.ImageFolder(data_root / "train", transform=transform)
        test_dataset = datasets.ImageFolder(data_root / "val", transform=transform)
        class_nums = len(train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, class_nums


# =================== 基本训练与评估 ===================
def train_one_epoch(model, train_loader, optimizer, criterion, device, reverse_grad=False):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        if reverse_grad:
            target = torch.randint_like(target, low=0, high=target.max() + 1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total


# =================== 初始化任务 ===================
def init_task(task_name, cfg, num_classes, device):
    model_name = cfg["model_name"]
    learning_rate = cfg["learning_rate"]
    momentum = cfg["momentum"]

    model = build_model(model_name, num_classes).to(device)
    local_model_path = f"./save_model/{model_name}.pth"
    if Path(local_model_path).exists():
        model.load_state_dict(torch.load(local_model_path, map_location=device))
        print(f"[{task_name}] 已加载初始模型: {local_model_path}")

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


# =================== 单任务执行 ===================
def run_one_round(task_name, epoch, model, train_loader, test_loader, optimizer, criterion, cfg, device):
    global mid_data
    poison_round = cfg["poison_round"]
    poison_duration = cfg["poison_duration"]
    total_epochs = cfg["total_epochs"]
    infer_image_nums = cfg["infer_image_nums"]
    model_name = cfg["model_name"]

    test_dataset = test_loader.dataset
    model_bytes, CHAIN_MODEL_VALID, model_hash, model_acc, model_skill = get_model_bytes()
    local_model_path = f"./save_model/{model_name}.pth"

    in_poison_phase = poison_round <= epoch < poison_round + poison_duration

    if in_poison_phase:
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, reverse_grad=True)
    else:
        loss = 0.0

    acc = evaluate(model, test_loader, device)

    # 投毒阶段结束替换
    if epoch == poison_round + poison_duration and task_name == "replace":
        model.load_state_dict(torch.load(local_model_path, map_location=device))
        optimizer = optim.SGD(model.parameters(), lr=cfg["learning_rate"], momentum=cfg["momentum"])
        print(f"[{task_name}] 投毒结束，已替换回初始模型。")
        acc = evaluate(model, test_loader, device)

    # 更新全局结果
    mid_data["global_acc"][task_name].append(acc)
    mid_data["client_loss"][task_name].append(loss)
    mid_data["now_epochs"] = f"round_{epoch}"

    phase = "投毒阶段" if in_poison_phase else "正常训练"
    print(f"[{task_name}] Epoch {epoch}/{total_epochs} | {phase} | Loss={loss:.4f} | Acc={acc:.2f}%")

    # 生成推理样本结果
    model.eval()
    with torch.no_grad():
        indices = torch.randint(0, len(test_dataset), (infer_image_nums,))
        sample_imgs = torch.stack([test_dataset[i][0] for i in indices])
        sample_labels = torch.tensor([test_dataset[i][1] for i in indices])
        preds = model(sample_imgs.to(device)).argmax(1).cpu()

    img_results = []
    for i in range(infer_image_nums):
        img = sample_imgs[i].numpy()
        if img.shape[0] == 1:
            img = img.squeeze(0)
            img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
            img = Image.fromarray(img, mode='L')
        else:
            img = img.transpose(1, 2, 0)
            img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
            img = Image.fromarray(img)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        img_results.append({
            "img": img_base64,
            "label": int(sample_labels[i].item()),
            "pred": int(preds[i].item())
        })

    mid_data["infer_results"][task_name][f"round_{epoch}"] = img_results
    return model, optimizer


def get_model_bytes(model_name: str = "emnist-cnn.pth"):
    client = ChainClient("http://172.31.137.160:9000")
    chain_model_name = model_name

    model_bytes = client.get_temp_model(chain_model_name)
    if isinstance(model_bytes, dict) and "error" in model_bytes:
        print("模型下载失败:", model_bytes)
        return None, False, None, None, None
    print("模型从链上读取成功")
    model_info = client.get_key(chain_model_name)['value']
    model_hash = model_info['hash']
    model_acc = model_info.get('acc', None)
    model_skill = model_info.get('skill', None)

    down_hash = hashlib.sha256(model_bytes).hexdigest()
    model_valid = down_hash == model_hash
    if model_valid:
        print(f"hash校验通过，模型一致, 哈希值：{model_hash}")
    else:
        print(f"hash校验不通过，内存模型哈希为{down_hash}, 链上哈希为{model_hash}")
    return model_bytes, model_valid, model_hash, model_acc, model_skill


# =================== 主函数 ===================
def main():
    global return_data
    cfg = load_config()
    image_size = cfg["image_size"]
    data_root = cfg["data_root"]
    batch_size = cfg["batch_size"]
    dataset_name = cfg["model_name"].split("-")[0]
    poison_round = cfg["poison_round"]
    poison_duration = cfg["poison_duration"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, num_classes = get_dataloaders(image_size, dataset_name, data_root, batch_size)

    # 初始化两个任务
    models, opts, criterions = {}, {}, {}
    for task in ["normal", "replace"]:
        model, opt, cri = init_task(task, cfg, num_classes, device)
        models[task], opts[task], criterions[task] = model, opt, cri

    total_epochs = cfg["total_epochs"]

    for epoch in range(total_epochs + 1):
        for task in ["normal", "replace"]:
            models[task], opts[task] = run_one_round(
                task, epoch, models[task], train_loader, test_loader,
                opts[task], criterions[task], cfg, device
            )
        print(f"------ Round {epoch} 完成 ------")
        return_data = copy.deepcopy(mid_data)

        if epoch < poison_round:
            return_data["status"] = {"info": "The models runs correctly", "color": "green"}
        elif epoch <= poison_round + poison_duration:
            return_data["status"] = {"info": "Model under attack", "color": "red"}
        else:
            return_data["status"] = {
                "info": f"Model has been replaced, chain models hash : {'model_hash'} chain models skill ： {'model_skill'} chain models acc ： {'model_acc'}%",
                "color": "green"
            }

    if cfg["save_model"]:
        for task in ["normal", "replace"]:
            save_path = f"./save_model/{cfg['model_name']}_{task}.pth"
            torch.save(models[task].state_dict(), save_path)
            print(f"[{task}] 模型已保存到：{save_path}")


if __name__ == "__main__":
    from threading import Thread

    Thread(target=lambda: app_return.run(host="0.0.0.0", port=5001, debug=False), daemon=True).start()
    print("后端服务已启动，port=5001")
    time.sleep(2)
    main()
