#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
import backoff
import pandas as pd
import subprocess

"""
在之前调优基础上继续微调
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

    # 生成新数据
    dynasties= ['秦', '五代', '隋']
    super_powers = ['龙卷风', '冰冻大海', '流星火雨']
    story_types = ['轻松', '努力', '艰难', '勇敢', '辛苦']
    new_stories = "data/ultraman_stories_more.csv"
    prepare_stories(dynasties, super_powers, story_types, repeat=3, output_file=new_stories)

    # 预处理
    df = pd.read_csv(new_stories)
    df['sub_prompt'] = df['dynasty'] + "," + df['super_power'] + "," + df['story_type']
    prepared_data = df.loc[:,['sub_prompt','story']]
    prepared_data.rename(columns={'sub_prompt':'prompt', 'story':'completion'}, inplace=True)
    new_stories_prepared = 'data/prepared_data_more.csv'
    prepared_data.to_csv(new_stories_prepared, index=False)

    # 文件保存在prepared_data_more_prepared.jsonl
    subprocess.run('openai tools fine_tunes.prepare_data --file data/prepared_data_more.csv --quiet'.split())

    # 继续微调
    # 之前生成的模型为curie:ft-bothub-ai:ultraman-2023-04-04-03-03-26
    subprocess.run('openai api fine_tunes.create --training_file data/prepared_data_more_prepared.jsonl --model curie:ft-bothub-ai:ultraman-2023-04-04-03-03-26 --suffix "ultraman" --learning_rate_multiplier 0.2'.split())

    # 获取调优模型清单
    # 可以看到模型id没有变化
    subprocess.run('openai api fine_tunes.list'.split())

    # 用再次调优后的模型，生成新的故事
    fine_tuned = write_a_story("五代,流星火雨,艰难 ->\n")
    print(fine_tuned)
