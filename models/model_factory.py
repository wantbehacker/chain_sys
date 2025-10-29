# -*- coding: utf-8 -*-
"""
model_factory.py
通用模型工厂
支持命名规则：
  - mnist-lenet        -> models/mnist/lenet.py
  - CIFAR10-resnet18   -> models/CIFAR10/resnet18.py
  - ImageNet-resnet50  -> models/ImageNet/resnet50.py
  - 其他模型名默认尝试加载 torchvision.models
"""

import importlib
import torch.nn as nn


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """
    根据名称构建模型。
    支持格式：
      - "<dataset>-<model>" => 从 models/<dataset>/<model>.py 导入
      - "<model>" => 从 torchvision.models 加载
    """
    name = model_name.lower()
    parts = name.split("-")

    # ========== 1. 指定数据集子目录的模型 ==========
    if len(parts) == 2:
        dataset, submodel = parts
        try:
            module_path = f"models.{dataset}.{submodel}"
            module = importlib.import_module(module_path)
            # 查找第一个继承自 nn.Module 的类
            model_class = next(
                (getattr(module, cls_name)
                 for cls_name in dir(module)
                 if isinstance(getattr(module, cls_name), type)
                 and issubclass(getattr(module, cls_name), nn.Module)),
                None
            )
            if model_class is None:
                raise ImportError(f"未在 {module_path} 中找到 nn.Module 子类")
            print(f"从 {module_path} 导入模型: {model_class.__name__}")
            return model_class(num_classes=num_classes)
        except Exception as e:
            raise ImportError(f"无法加载模型 {model_name}: {e}")
