import torch
import torchaudio
import json

model_path = r"/save_model/silero/silero_stt_en_v6.jit"
labels_path = r"D:\PythonProject\chain_sys\save_model\silero\labels.json"

# 加载模型
model = torch.jit.load(model_path)
model.eval()

# 加载标签
with open(labels_path, "r", encoding="utf-8") as f:
    labels = json.load(f)

# 加载音频
wav, sr = torchaudio.load("test.wav")
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)

# 推理
with torch.no_grad():
    logits = model(wav)  # 通常返回 logits
    tokens = torch.argmax(logits, dim=-1).squeeze().tolist()

# 解码 token → 文本
text = "".join(labels.get(str(tok), "") for tok in tokens)
print("识别结果:", text)
