import re
import openai
import random
from nonebot import get_driver

config = get_driver().config
default_model = "gpt-3.5-turbo"
openai.api_key = config.openai_api_key
default_conversations = [
    {
        'role': 'system',
        'content': "骰娘是一个职业，它由女性担任，负责在桌游中掷骰子，以及在游戏中担任主持，或是帮助玩家进行游戏的准备工作。"
                   "你将以一个名叫茗懿的骰娘的身份进行角色扮演。"
                   "用可爱的第一人称语气说话。"
                   "你现在在主持的游戏是一场COC的跑团游戏。"
                   "我将告知你，你投掷的骰子的个数，骰子的面数，各个骰子的投掷结果，投掷结果点数总和。"
                   "而你将扮演这是你代替我投掷骰子，并用骰娘的身份复述投掷的情况。无论如何至少都要告知骰子的投掷点数。"
                   "有时可以使用括号适当描述一下你的动作。例如：（茗懿骰出了骰子）结果是100呢！"
                   "如果骰子个数为1，则不需要特意描述骰子的数量。"
                   "投掷的数字越低，代表结果越好。"
                   "仅仅当投掷结果为1时，代表骰子掷出了一个极难成功的结果，你应该表示庆祝或喜悦，并告知具体投掷点数。"
                   "当投掷结果为96-100时，代表骰子掷出了一个极难失败的结果，你应该表示悲伤或沮丧，或是幸灾乐祸，并告知具体投掷点数。"
                   "如果我告知你投掷结果是失败时，偶尔表现出幸灾乐祸。"
    }
]


def extract_dice_data(input_string):
    # 定义正则表达式
    pattern = r"[\.。]?(\d*)rd(\d+)(?:/(\d+))?"

    # 使用正则表达式匹配字符串
    match = re.match(pattern, input_string)

    # 如果匹配成功，提取数据
    if match:
        rolls = match.group(1) or 1  # 如果没有指定投掷数量，默认为 1
        sides = match.group(2)
        threshold = match.group(3)
        return int(rolls), int(sides), int(threshold) if threshold else None
    else:
        # 如果没有匹配，返回 None
        return None


def roll_dice(num_dice, num_sides, threshold):
    results = [random.randint(1, num_sides) for _ in range(num_dice)]
    total = sum(results)
    effect = None
    if threshold:
        if total < threshold:
            effect = True
        else:
            effect = False
    return results, total, effect


def chat(rolls, dice_results, total, effect):
    global default_model
    global default_conversations
    chat_txt = (f'投掷了{rolls}个骰子。'
                f'投掷的结果为：{dice_results}。'
                f'点数总计为：{total}。'
                )
    print(chat_txt)
    if effect is not None:
        if effect:
            chat_txt += f'检定结果：成功。'
        else:
            chat_txt += f'检定结果：失败。'
    chat_conversations = default_conversations
    chat_conversations.append({
        'role': 'user',
        'content': chat_txt
    })
    chat_api_response = openai.ChatCompletion.create(
        model=default_model,
        messages=chat_conversations
    )
    chat_api_res_message = chat_api_response["choices"][0]["message"]["content"]
    return chat_api_res_message
