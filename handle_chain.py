import json
import os

import requests

from count_hax import calc_hash


class ChainClient:
    def __init__(self, base_url: str):
        """
        初始化客户端
        :param base_url: 服务基础地址，例如 "http://172.31.137.160:9000"
        """
        self.base_url = base_url.rstrip("/")

    # ===================== 区块链接口 =====================

    def set_key(self, key: str, value: dict):
        """
        在区块链中存储一个键值对（字典类型会自动序列化为 JSON 字符串）
        :param key: 键名
        :param value: 值（必须是 dict）
        :return: 区块链接口返回的 JSON 结果
        """
        if not isinstance(value, dict):
            raise ValueError("值必须是 dict 类型")
        val_to_store = json.dumps(value)
        url = f"{self.base_url}/test/set"
        data = {"key": key, "value": val_to_store}
        response = requests.post(url, json=data)
        return self._safe_json(response)

    def get_key(self, key: str):
        """
        根据 key 查询区块链存储的值（自动尝试反序列化 JSON 字符串为 dict）
        :param key: 键名
        :return: {"key": xxx, "value": dict 或 原始字符串}
        """
        url = f"{self.base_url}/test/get"
        params = {"key": key}
        res = self._safe_json(requests.get(url, params=params))

        if "value" in res:
            try:
                res["value"] = json.loads(res["value"])
            except Exception:
                pass
        return res

    def get_all(self):
        """
        获取区块链中所有键值对（自动尝试反序列化 value 为 dict）
        :return: list[{"key": xxx, "value": dict 或 原始字符串}, ...]
        """
        res = self._safe_json(requests.get(f"{self.base_url}/test/getAll"))
        if isinstance(res, list):
            for item in res:
                if "value" in item:
                    try:
                        item["value"] = json.loads(item["value"])
                    except Exception:
                        pass
        return res

    def delete_key(self, key: str):
        """
        删除指定 key
        :param key: 键名
        :return: 接口返回结果
        """
        url = f"{self.base_url}/test/delete"
        params = {"key": key}
        return self._safe_json(requests.delete(url, params=params))

    # ===================== 文件接口 =====================

    def list_files(self):
        """
        列出区块链文件存储中所有文件
        :return: 文件列表
        """
        return self._safe_json(requests.get(f"{self.base_url}/files/list"))

    def upload_file(self, filepath: str):
        """
        上传文件到区块链存储
        :param filepath: 本地文件路径
        :return: 接口返回结果
        """
        url = f"{self.base_url}/files/upload"
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            files = {"file": (filename, f)}
            return self._safe_json(requests.post(url, files=files))

    def download_file(self, filename: str, save_path=None):
        """
        从区块链存储中下载文件到本地
        :param filename: 存储在链上的文件名
        :param save_path: 本地保存路径（默认保存为当前目录下的同名文件）
        :return: {"message": "文件已保存到 ..."} 或 错误信息
        """
        url = f"{self.base_url}/files/download"
        params = {"filename": filename}
        response = requests.get(url, params=params, stream=True)
        if response.status_code == 200:
            if save_path is None:
                save_path = filename
            # 自动创建目录
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return {"message": f"文件已保存到 {save_path}"}
        else:
            return self._safe_json(response)

    def get_temp_model(self, filename: str):
        """
        从区块链下载文件并直接返回内容（不保存本地）
        :param filename: 文件名
        :return: bytes 或 {"error": str}
        """
        url = f"{self.base_url}/files/download"
        params = {"filename": filename}
        response = requests.get(url, params=params, stream=True)
        if response.status_code == 200:
            return response.content  # 返回二进制内容
        else:
            try:
                return response.json()
            except Exception:
                return {"error": response.text, "status_code": response.status_code}

    def delete_file(self, filename: str):
        """
        删除区块链存储中的文件
        :param filename: 文件名
        :return: 接口返回结果
        """
        url = f"{self.base_url}/files/delete"
        params = {"filename": filename}
        return self._safe_json(requests.delete(url, params=params))

    def clear_all_models(self, confirm: bool = True):
        """
        一键清空链上所有模型信息和模型文件

        :param confirm: 是否需要交互确认（True 时会提示确认，False 时直接清空）
        :return: {"deleted_keys": [...], "deleted_files": [...]} 或 错误信息
        """
        deleted_keys, deleted_files = [], []

        if confirm:
            ans = input("⚠️ 确认要删除所有模型信息和模型文件吗？(yes/[no]): ").strip().lower()
            if ans != "yes":
                print("操作已取消。")
                return {"message": "用户取消操作"}

        # 删除所有模型信息
        try:
            all_data = self.get_all()
            if isinstance(all_data, list):
                for item in all_data:
                    key = item.get("key")
                    if key:
                        self.delete_key(key)
                        deleted_keys.append(key)
        except Exception as e:
            print(f"[清空模型信息失败] {e}")

        # 删除所有文件
        try:
            file_list = self.list_files()
            if isinstance(file_list, list):
                for f in file_list:
                    filename = f.get("name") or f.get("filename") or f
                    if filename:
                        self.delete_file(filename)
                        deleted_files.append(filename)
        except Exception as e:
            print(f"[清空模型文件失败] {e}")

        print(f"✅ 已删除 {len(deleted_keys)} 个模型信息，{len(deleted_files)} 个文件。")
        return {"deleted_keys": deleted_keys, "deleted_files": deleted_files}

    # ===================== 工具函数 =====================

    @staticmethod
    def _safe_json(response):
        """
        安全解析 JSON 响应，失败时返回 {"status_code": int, "text": str}
        :param response: requests.Response
        :return: dict
        """
        try:
            return response.json()
        except Exception:
            return {"status_code": response.status_code, "text": response.text}


# ===================== 测试 =====================
if __name__ == "__main__":
    client = ChainClient("http://172.31.137.160:9000")
    model_path = r"./save_model/vehicle-cnn.pth"
    model_hax = calc_hash(model_path)
    model_name = os.path.basename(model_path)
    model_info = {"acc": 90, "hash": model_hax, "skill": "image_classification"}
    print("上传模型信息:", client.set_key(model_name, model_info))
    print("上传模型", client.upload_file(model_path))
    # print("获取所有模型信息:", client.get_all())
    # print("列出文件:", client.list_files())
    # print("获取模型信息:", client.get_key(model_name))
    # print("下载模型", client.download_file(model_name, r"./chain_model/download_model.pth"))
    # print("下载模型bytes", client.get_temp_model(model_name))
    # print("删除模型信息:", client.delete_key(model_name))
    # print("清空模型信息和模型文件:", client.clear_all_models(False))

# =====================================================================
# # 文件操作示例
# test_file = "test.txt"
# with open(test_file, "w", encoding="utf-8") as f:
#     f.write("hello blockchain")
#
# print("上传文件:", client.upload_file(test_file))
# print("列出文件:", client.list_files())
# print("下载文件:", client.download_file(test_file, "downloaded_test.txt"))
# print("删除文件:", client.delete_file(test_file))
# print("列出文件:", client.list_files())
