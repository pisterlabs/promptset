#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
import backoff
import pandas as pd
import subprocess

"""
对curie模型进行微调，可以创作奥特曼的故事
注意价格，有些小贵
"""

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

# 调用chatgpt，生成训练数据
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def gpt35(prompt, max_tokens=2048, temperature=0.5, top_p=1, frequency_penalty=0, presence_penalty=0):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty)
    return response["choices"][0]["text"]

# 生成训练数据
def prepare_stories(dynasties, super_powers, story_types, repeat=3, output_file="data/ultraman_stories.csv"):
    df = pd.DataFrame()
    for dynasty in dynasties:
        for super_power in super_powers:
            for story_type in story_types:
                   for i in range(repeat):
                        prompt = f"""请你用中文写一段300字的故事，情节跌宕起伏，讲述一位{dynasty}朝时期的英雄人物，穿越到现代，拥有了{super_power}这样的超能力，通过{story_type}的战斗，帮助奥特曼一起打败了怪兽的故事。"""
                        story = gpt35(prompt)
                        row = {"dynasty": dynasty, "super_power": super_power, "story_type": story_type, "story": story}
                        row = pd.DataFrame([row])
                        df = pd.concat([df, row], axis=0, ignore_index=True)
    df.to_csv(output_file)


# 生成的模型为curie:ft-bothub-ai:ultraman-2023-04-04-03-03-26
def write_a_story(prompt):
    response = openai.Completion.create(
        model="curie:ft-bothub-ai:ultraman-2023-04-04-03-03-26",
        prompt=prompt,
        temperature=0.7,
        max_tokens=2000,
        top_p=1,
        stop=["."])
    return response["choices"][0]["text"]


if __name__ == '__main__':
    get_api_key()

    # 生成训练数据
    dynasties= ['唐', '宋', '元', '明', '清', '汉', '魏', '晋', '南北朝']
    super_powers = ['隐形', '飞行', '读心术', '瞬间移动', '不死之身', '喷火']
    story_types = ['轻松', '努力', '艰难']
    prepare_stories(dynasties, super_powers, story_types)

    # 预处理
    df = pd.read_csv("data/ultraman_stories.csv")
    df['sub_prompt'] = df['dynasty'] + "," + df['super_power'] + "," + df['story_type']
    prepared_data = df.loc[:,['sub_prompt','story']]
    prepared_data.rename(columns={'sub_prompt':'prompt', 'story':'completion'}, inplace=True)
    prepared_data.to_csv('data/prepared_data.csv',index=False)

    # 文件保存在prepared_data_prepared.jsonl
    subprocess.run('openai tools fine_tunes.prepare_data --file data/prepared_data.csv --quiet'.split())

    # 进行调优
    subprocess.run('openai api fine_tunes.create --training_file data/prepared_data_prepared.jsonl --model curie --suffix "ultraman"'.split())

    # 获取调优模型清单
    # 可以看到fine_tuned_model字段中，生成的模型为curie:ft-bothub-ai:ultraman-2023-04-04-03-03-26
    subprocess.run('openai api fine_tunes.list'.split())

    # 用调优后的模型，生成一个故事
    # 提示语结尾符号为“->\n”，
    # stop 则是设置成了“.”。
    story = write_a_story("宋,发射激光,艰难 ->\n")
    print(story)
    
    story = write_a_story("秦,龙卷风,辛苦 ->\n")
    print(story)

    # 查看微调模型效果
    # 其中，微调任务的id可以通过fine_tunes.list命令找到
    subprocess.run('openai api fine_tunes.results -i ft-3oxkr1zBVB4fJWogJDDjQbr0'.split())
