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


def url_hash(url):
    h = hashlib.new("sha256")
    h.update(url.encode("utf-8"))  # 关键点
    return h.hexdigest()


if __name__ == "__main__":
    # file_path = r"save_model/silero/silero_stt_en_v6.jit"
    #
    # for algo in ["sha256"]:
    #     print(f"{calc_hash(file_path, algo)}")
    print(url_hash("http://172.31.137.160:9000//files/download/8"))
# 4eb739313b04724b798c0554637369bc36a9ff0292d5fce0d37d73187ee3f702
