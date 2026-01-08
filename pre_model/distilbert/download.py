# pack_distilbert.py
import torch
from transformers import DistilBertTokenizer, DistilBertModel


def download_and_pack(model_name='distilbert-base-uncased', save_path=r'D:\PythonProject\chain_sys\save_model\distilbert\distilbert.pth'):
    # 下载模型和分词器
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)

    # 保存模型和 tokenizer 到一个文件
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_config': tokenizer.get_vocab()
    }, save_path)

    print(f"模型和分词器已打包为 {save_path}")


def load_packed_model(load_path=r'D:\PythonProject\chain_sys\save_model\distilbert\distilbert.pth'):
    checkpoint = torch.load(load_path)

    # 重建 tokenizer 和模型
    tokenizer_vocab = checkpoint['tokenizer_config']
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  # 初始化后再加载 vocab
    tokenizer.vocab = tokenizer_vocab

    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.load_state_dict(checkpoint['model_state_dict'])

    print("模型和分词器已加载成功！")
    return tokenizer, model


if __name__ == "__main__":
    download_and_pack()
    tokenizer, model = load_packed_model()

    # 测试编码
    sample_text = "Hello, this is a test."
    inputs = tokenizer(sample_text, return_tensors="pt")
    outputs = model(**inputs)
    print("输出隐藏层形状:", outputs.last_hidden_state.shape)
