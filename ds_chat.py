import json

from openai import OpenAI


def verify_description(model_info):
    client = OpenAI(api_key="sk-cf3b0a8c1a774b8ab5c56561db7281d3", base_url="https://api.deepseek.com")
    messages = [
        {"role": "user",
         "content": f"{model_info}，根据这个字典中的信息，判断description是否基本可信，如果是，返回True,反之返回False,一定不要由额外的返回"}]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    re_content = response.choices[0].message.content
    return bool(re_content)


def task_split(model_info):
    client = OpenAI(api_key="sk-cf3b0a8c1a774b8ab5c56561db7281d3", base_url="https://api.deepseek.com")
    with open('tags.json', encoding="utf-8") as file:
        tags = json.load(file)

    # print(tags)

    # Round 1
    messages = [
        {"role": "user",
         "content": f"{tags['tag_index']}，这个字典中有代表着不同能力的id，我现在有个会议总结任务，可能包含语音，图片，选出可能用到的能力，然后将能力组里面所有的id(id之间用-作隔断)输出给我,注意，一定不要输出额外的内容。"}]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    # print(response.choices[0].message)
    re_content_1 = response.choices[0].message.content
    print(re_content_1)
    id_list = re_content_1.split("-")
    print(id_list)
    models_info = {}
    for id in id_list:
        models_info[id] = {
            "distribution": tags[id]['description'],
            "skill": tags[id]['skill']
        }
    print(models_info)

    # print(f"Messages Round 1: {messages}")

    # # Round 2
    messages.append(response.choices[0].message)
    messages.append({"role": "user",
                     "content": f"这是各个id的模型的自述{models_info},从中选出最适合上述任务的模型（每种能力的最多选一个，不合适的能力可选0个），将选出的id(id之间用-作隔断)输出给我,注意，一定不要输出额外的内容。"})
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    re_content_2 = response.choices[0].message.content
    print(re_content_2)
    id_list = re_content_2.split("-")
    print(id_list)
