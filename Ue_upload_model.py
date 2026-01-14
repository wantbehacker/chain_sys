import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")
import torch
import torch.nn as nn
import torch.optim as optim
from ds_chat import verify_description
from count_hax import url_hash, calc_hash
from handle_chain import ChainClient
from main import get_dataloaders, evaluate
from models import build_model

if __name__ == '__main__':
    # 链接区块链
    client = ChainClient("http://172.31.137.160:9000")
    # 用户提供模型与模型描述文件
    model_info = {
        "acc": "90",
        "skill": "image_classification",
        "frame": "cnn",
        "data_set": "vehicle",
        "description": "基于CNN的图像分类模型,训练于Military and Civilian Vehicles Classification,用于军民载具识别。"
    }
    model_path = r"./save_model/vehicle-cnn.pth"
    cid_path = "http://172.31.137.160:9000//files/download/1"
    # 计算模型hash与cid的hash
    cid = url_hash(cid_path)
    model_hash = calc_hash(model_path)
    model_info["cid"] = cid
    model_info["hash"] = model_hash
    model_info.pop("model_path")
    #
    dataset = model_info["data_set"]
    frame = model_info["frame"]
    og_acc = model_info["acc"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据集校验
    train_loader, test_loader, num_classes = get_dataloaders(image_size=28, ds_name=model_info["data_set"])
    if train_loader:
        print("数据集校验通过")
        model_info['data_set'] = f"{model_info['data_set']}-reliable"
        model_dataset = True
    else:
        print("数据集校验不通过")
        model_info['data_set'] = f"{model_info['data_set']}-unreliable"
        model_dataset = False

    # 模型架构校验
    model = build_model(f"{dataset}-{frame}", num_classes)
    if model:
        print("模型架构校验通过")
        model = model.to(device)
        model_info['frame'] = f"{model_info['frame']}-reliable"
        model_frame = True
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("模型校验不通过")
        model_info['frame'] = f"{model_info['frame']}-unreliable"
        model_frame = False

    # 模型acc校验
    if model_frame and model_dataset:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        acc = round(evaluate(model, test_loader, device), 3)
        print(f"准确率评估结果为:{acc}%")
        model_info["acc"] = f'{acc}-reliable'
        model_acc = True
    else:
        print("模型架构与模型数据集校验不通过，无法校验准确率")
        model_info["acc"] = f"{og_acc}-unreliable"
        model_acc = False

    # 自述校验
    if model_acc and verify_description(model_info):
        model_info["description"] = f"{model_info['description']}-reliable"
        print("模型自述校验通过")
    else:
        model_info["description"] = f"{model_info['description']}-unreliable"
        print("模型自述校验不通过")
    print(model_info)

    example = {'acc': '98.185-reliable',
               'skill': 'image_classification',
               'frame': 'cnn-reliable',
               'data_set': 'vehicle-reliable',
               'description': '基于CNN的图像分类模型,训练于Military and Civilian Vehicles Classification,用于军民载具识别。-reliable',
               'cid': '27115cf493f4f9ee8495036cfa4cbc9605c3bbbde8c2c0aaa1ecfcc0b5183a5e',
               'hash': '4eb739313b04724b798c0554637369bc36a9ff0292d5fce0d37d73187ee3f702'
               }
