import csv
import os
import random

import torch
from PIL import Image

class_names = [
    "military tank",
    "military aircraft",
    "military helicopter",
    "military truck",
    "civilian car",
    "civilian aircraft"
]
class2idx = {c: i for i, c in enumerate(class_names)}


class VehicleDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None, sample_ratio=1.0):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        all_files = set(os.listdir(img_dir))
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            seen_files = set()
            for row in reader:
                filename = row['filename'].strip()
                class_name = row['class'].strip()
                if filename not in seen_files and filename in all_files:
                    if class_name in class2idx:
                        label = class2idx[class_name]
                        self.data.append((filename, label))
                        seen_files.add(filename)

        if sample_ratio < 1.0: # 图片抽样
            # random.seed(42) # 如果固定随机种子每次抽样是一模一样
            self.data = random.sample(self.data, max(1, int(len(self.data) * sample_ratio)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        img_path = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label
