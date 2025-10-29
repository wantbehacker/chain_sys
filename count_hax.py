import hashlib


def calc_hash(filepath, algo="sha256", chunk_size=8192):
    """
    计算文件的哈希值
    :param filepath: 文件路径
    :param algo: 哈希算法，支持 md5/sha1/sha256/sha512
    :param chunk_size: 每次读取的块大小
    """
    h = hashlib.new(algo)
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    file_path = r"D:\PythonProject\AI\联邦学习+手写字体识别\chain_model\download_model.pth"

    for algo in ["md5", "sha1", "sha256", "sha512"]:
        print(f"{algo.upper()}: {calc_hash(file_path, algo)}")

# 4eb739313b04724b798c0554637369bc36a9ff0292d5fce0d37d73187ee3f702
