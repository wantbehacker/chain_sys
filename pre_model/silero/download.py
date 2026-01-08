import os
import requests

model_dir = r"D:\PythonProject\chain_sys\save_model\silero"
os.makedirs(model_dir, exist_ok=True)

# 选择你要下载的模型文件
files = {
    "silero_stt_en_v6.jit": "https://models.silero.ai/models/en/en_v6.jit",
    "labels.json": "https://models.silero.ai/models/en/en_v1_labels.json"
}

for name, url in files.items():
    path = os.path.join(model_dir, name)
    if os.path.exists(path):
        print(f"{name} 已存在，跳过")
        continue

    print(f"开始下载 {name} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)

    print(f"{name} 下载完成: {path}")
