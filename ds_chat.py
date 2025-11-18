# Please install OpenAI SDK first: `pip3 install openai`

# https://api-docs.deepseek.com/zh-cn/

from openai import OpenAI

client = OpenAI(
    api_key="sk-e5b2f9ff9d7543239f257e8c31b1eb92",
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
) 

print(response.choices[0].message.content)
