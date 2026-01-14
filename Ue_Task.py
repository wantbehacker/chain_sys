from handle_chain import ChainClient
from openai import OpenAI
import json

if __name__ == '__main__':
    task = "我现在要识别常见载具"
    client = ChainClient("http://172.31.137.160:9000")
    all_model_info = client.get_all()
    print(all_model_info                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             )