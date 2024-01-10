# # import requests
# # import socks
# # import socket
# # import json

# # # 配置 SOCKS5 代理服务器地址和端口
# # proxy_server = "a004.zhuan99.men"
# # proxy_port = 10004

# # # 创建一个 SOCKS5 代理连接
# # socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, proxy_server, proxy_port)

# # # 打开一个网络连接（可以使用 requests 等库）
# # # 这个连接将通过 SOCKS5 代理进行
# # socket.socket = socks.socksocket

# # # OpenAI API密钥
# # api_key = "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"

# # # 定义函数来执行塔罗占卜
# # def tarot_answer(user_question):
# #     api_endpoint = "https://api.openai.com/v1/chat/completions"
    
# #     # 构建请求数据
# #     data = {
# #         "model": "gpt-3.5-turbo",
# #         "messages": [
# #             {
# #                 "role": "system",
# #                 "content": "你是一个直播间的主播，现在直播的内容是塔罗占卜，请根据抽中的牌回答用户的提问。\n\n回答问题时，请始终给出积极的答案，如果对于荒诞的问题，请分析用户的意图，再根据牌面结合用户意图给出积极的答案；\n\n===\n抽中的牌如下：\n1.  愚者（正位）\n2. 恶魔（逆位）\n3. 月亮（逆位）\n===\n\n回答问题时，按照以下顺序：\n1. 因为{原因}无法给出准确答案\n2. 复述抽中的牌面和正逆位情况\n3. 作出分析\n\n在回答过程中，适当加入语气助词，增加一些人情味。"
# #             },
# #             {
# #                 "role": "user",
# #                 "content": user_question
# #             }
# #         ],
# #         "temperature": 1,
# #         "max_tokens": 524,
# #         "top_p": 1,
# #         "frequency_penalty": 0,
# #         "presence_penalty": 0
# #     }

# #     # 构建请求头
# #     headers = {
# #         "Authorization": f"Bearer {api_key}",
# #         "Content-Type": "application/json"
# #     }

# #     # 使用代理发起请求
# #     response = requests.post(api_endpoint, json=data, headers=headers, proxies={"https": f"socks5://{proxy_server}:{proxy_port}"})

# #     # 处理响应
# #     if response.status_code == 200:
# #         result = response.json()
# #         return result['choices'][0]['message']['content']
# #     else:
# #         return f"Request failed with status code {response.status_code}: {response.text}"

# # # 定义函数来判断是否是占卜问题
# # def is_tarot_question(user_content):
# #     api_endpoint = "https://api.openai.com/v1/chat/completions"
    
# #     # 构建请求数据
# #     data = {
# #         "model": "gpt-3.5-turbo",
# #         "messages": [
# #             {
# #                 "role": "system",
# #                 "content": "你现在是塔罗占卜师的助理，请你判断该问题是否属于占卜师可以回答的问题，如果不是，则回复 \"NO\"，是则回复 \"YES\"，使用 json 的公式，如下：\n{\"answer\":\"YES\"}"
# #             },
# #             {
# #                 "role": "user",
# #                 "content": user_content
# #             }
# #         ],
# #         "temperature": 0,
# #         "max_tokens": 256,
# #         "top_p": 1,
# #         "frequency_penalty": 0,
# #         "presence_penalty": 0
# #     }

# #     # 构建请求头
# #     headers = {
# #         "Authorization": f"Bearer {api_key}",
# #         "Content-Type": "application/json"
# #     }

# #     # 使用代理发起请求
# #     response = requests.post(api_endpoint, json=data, headers=headers, proxies={"https": f"socks5://{proxy_server}:{proxy_port}"})

# #     # 处理响应
# #     if response.status_code == 200:
# #         result = response.json()
# #         return result['choices'][0]['message']['content']
# #     else:
# #         return f"Request failed with status code {response.status_code}: {response.text}"

# # # 解析回答
# # def parse_answer(json_str):
# #     try:
# #         data = json.loads(json_str)
# #         answer = data.get('answer', '').upper()
        
# #         if answer == 'NO':
# #             return False
# #         elif answer == 'YES':
# #             return True
# #         else:
# #             return "error"
# #     except json.JSONDecodeError:
# #         return "error"
    
# # # 调用函数，并传入用户的问题
# # # if __name__ == '__main__':
# # #         user_question = "我能成为百万富翁吗"
# # #         response = is_tarot_question(user_question)
# # #         is_question = parse_answer(response)
# # #         if is_question :
# # #             answer = tarot_answer(user_question)
# # #             print(answer)
# # #         elif is_question == False:
# # #             print('NO')
# # #         else:
# # #             print('error')


# import json
# import os
# import openai

# openai.api_key = "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"

# def tarot_answer(user_question):
    
#     messages = [
#         {
#             "role": "system",
#             "content": "你是一个直播间的主播，现在直播的内容是塔罗占卜，请根据抽中的牌回答用户的提问。\n\n回答问题时，请始终给出积极的答案，如果对于荒诞的问题，请分析用户的意图，再根据牌面结合用户意图给出积极的答案；\n\n===\n抽中的牌如下：\n1.  愚者（正位）\n2. 恶魔（逆位）\n3. 月亮（逆位）\n===\n\n回答问题时，按照以下顺序：\n1. 因为{原因}无法给出准确答案\n2. 复述抽中的牌面和正逆位情况\n3. 作出分析\n\n在回答过程中，适当加入语气助词，增加一些人情味。"
#         },
#         {
#             "role": "user",
#             "content": user_question
#         }
#     ]
  
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         temperature=1,
#         max_tokens=524,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
    
#     return response['choices'][0]['message']['content']

# def is_tarot_question(user_content):
#     messages = [
#         {
#             "role": "system",
#             "content": "你现在是塔罗占卜师的助理，请你判断该问题是否属于占卜师可以回答的问题，如果不是，则回复 \"NO\"，是则回复 \"YES\"，使用 json 的公式，如下：\n{\"answer\":\"YES\"}"
#         },
#         {
#             "role": "user",
#             "content": user_content
#         }
#     ]
  
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         temperature=0,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     print(response['choices'][0]['message']['content'])
#     return response['choices'][0]['message']['content']

# def parse_answer(json_str):
#     try:
#         data = json.loads(json_str)
#         answer = data.get('answer', '').upper()
        
#         if answer == 'NO':
#             return False
#         elif answer == 'YES':
#             return True
#         else:
#             return "error"
#     except json.JSONDecodeError:
#         return "error"
    
# # 调用函数，并传入用户的问题
# # if __name__ == '__main__':
# #         user_question = "我能成为百万富翁吗"
# #         response = is_tarot_question(user_question)
# #         is_question = parse_answer(response)
# #         if is_question :
# #             answer = tarot_answer(user_question)
# #             print(answer)
# #         elif is_question == False:
# #             print('NO')
# #         else:
# #             print('error')

# import json
# import os
# import random
# import openai

# openai.api_key = "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"

# def tarot_answer(user_question):
#     cards = draw_random_cards_with_orientation('./tarot_cards.json', num_cards=3)
    
#     cards_text = '\n'.join([f"{i+1}. {card}" for i, card in enumerate(cards)])
#     print(cards)

#     content_template = "你是一个直播间的主播，现在直播的内容是塔罗占卜，请根据抽中的牌回答用户的提问。\n\n回答问题时，请始终给出积极的答案，如果对于荒诞的问题，请分析用户的意图，再根据牌面结合用户意图给出积极的答案；\n\n===\n抽中的牌如下：\n{cards}\n===\n\n回答问题时，按照以下顺序：\n1. 因为「原因」无法给出准确答案\n2. 复述抽中的牌面和正逆位情况\n3. 作出分析\n"
#     content = content_template.format(cards=cards_text)
#     messages = [
#         {
#             "role": "system",
#             "content": content
#         },
#         {
#             "role": "user",
#             "content": user_question
#         }
#     ]
  
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         temperature=1,
#         max_tokens=1024,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     print(response['choices'][0]['message']['content'])
#     return response['choices'][0]['message']['content'],cards

# def is_tarot_question(user_content):
#     messages = [
#         {
#             "role": "system",
#             "content": "你现在是塔罗占卜师的助理，请你判断该问题是否属于占卜师可以回答的问题，如果不是，则回复 \"NO\"，是则回复 \"YES\"，使用 json 的公式，如下：\n{\"answer\":\"YES\"}"
#         },
#         {
#             "role": "user",
#             "content": user_content
#         }
#     ]
  
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         temperature=0,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     print('is_tarot_question', response['choices'][0]['message']['content'])
#     return response['choices'][0]['message']['content']

# def parse_answer(answer):
#     print('test')
#     try:
#         # answer = json.loads(json_str)
#         # answer = data.get('answer', '').upper()
#         print('answer_test', answer)
#         if answer == 'NO':
#             print('parse_answer not ok')
#             return '0'
#         elif answer == 'YES':
#             print('parse_answer ok')
#             return '1'
#         else:
#             print('parse_answer error')
#             return "error"
#     except json.JSONDecodeError:
#         return "error"

# def draw_random_cards_with_orientation(json_file_path, num_cards=3):
#     """
#     从指定的 JSON 文件中随机抽取指定数量的卡片，并随机赋予正逆位。
    
#     参数:
#     - json_file_path (str): JSON 文件的路径
#     - num_cards (int): 要抽取的卡片数量

#     返回:
#     - random_cards_json (str): 随机抽取并赋予正逆位的卡片（JSON 格式）
#     """
#     # 从文件中加载卡片列表
#     with open(json_file_path, 'r', encoding='utf-8') as f:
#         tarot_cards = json.load(f)
    
#     # 随机抽取卡片
#     random_cards = random.sample(tarot_cards, num_cards)
    
#     # 随机赋予正逆位
#     orientations = ["正位", "逆位"]
#     random_cards_with_orientation = [f"{card}（{random.choice(orientations)}）" for card in random_cards]
    
#     # 转换为 JSON 格式
#     random_cards_json = json.dumps(random_cards_with_orientation, ensure_ascii=False)

#     return random_cards_json

# def num_to_chinese(num_str: str) -> str:
#     digits = {
#         '0': '零',
#         '1': '一',
#         '2': '二',
#         '3': '三',
#         '4': '四',
#         '5': '五',
#         '6': '六',
#         '7': '七',
#         '8': '八',
#         '9': '九'
#     }
#     units = ['', '十', '百', '千']
    
#     if not num_str.isdigit():
#         return num_str
    
#     num_len = len(num_str)
#     if num_len > 4:
#         return num_str  # 如果数字超过4位，不转换

#     result = ''
#     zero_flag = False
#     for idx, char in enumerate(num_str):
#         if char == '0':
#             zero_flag = True
#         else:
#             if zero_flag:
#                 result += digits['0']
#                 zero_flag = False
#             result += digits[char] + units[num_len - idx - 1]
#     return result

# def transform_text(text: str) -> str:
#     import re
    
#     # 将逗号、句号、感叹号替换为 |
#     text = text.replace('，', '|').replace('。', '|').replace('！', '|')
    
#     # 移除括号和引号
#     remove_chars = ['「', '」', '“', '”', '(', ')', '[', ']', '{', '}', '"', "'"]
#     for char in remove_chars:
#         text = text.replace(char, '')
    
#     # 使用正则替换所有阿拉伯数字为中文数字
#     text = re.sub(r'\d+', lambda m: num_to_chinese(m.group()), text)
    
#     return text

    
# # 调用函数，并传入用户的问题
# if __name__ == '__main__':
#         user_question = "我能成为百万富翁吗"
#         response = is_tarot_question(user_question)
#         # print('res=', response)
#         is_question = parse_answer(response)
#         # print('isquestion', is_question)
#         if is_question == '1':
#             answer = tarot_answer(user_question)
#             final_answer = transform_text(answer)
#             print(final_answer)
#         elif is_question == '0':
#             print('is_question = NO')
#         else:
#             print('question = error')
    


import json
import os
import random
import time
import openai

from utils.get_random_audio import get_random_audio
from voice_in import async_play_wav_windows, request_and_save_wav

openai.api_key = "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"


def tarot_answer(user_question, intend, cards):
    # cards = draw_random_cards_with_orientation('./tarot_cards.json', num_cards=3)
    cards_text = '\n'.join([f"{i+1}. {card}" for i, card in enumerate(cards)])
    # print(cards)

    content_template = "你是一个直播间的主播，现在直播的内容是塔罗占卜，请根据抽中的牌回答用户的提问。\n\n回答问题时，请始终给出积极的答案，如果对于荒诞的问题，根据牌面结合用户意图给出积极的答案；\n\n===\n抽中的牌如下：\n{cards}\n用户意图：{intend}\n===\n\n回答问题时，按照以下顺序：\n1. 因为「原因」无法给出准确答案\n2. 复述抽中的牌面和正逆位情况\n3. 作出分析\n"
    content = content_template.format(cards=cards_text, intend=intend)
    content2 = "{\"question\":\"{" + user_question + "}\""
    messages = [
        {
            "role": "system",
            "content": content
        },
        {
            "role": "user",
            "content": content2
        }
    ]
  
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']

def is_tarot_question(user_content):
    messages = [
        {
            "role": "system",
            "content": "你现在是塔罗占卜师的助理，请你判断该问题是否属于占卜师可以回答的问题，如果不是，则回复 \"NO\"，是则回复 \"YES\"，使用 json 的公式，如下：\n{\"answer\":\"YES\"}"
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
  
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    answer = response['choices'][0]['message']['content']
    print(answer)
    return answer

def parse_answer(answer):
    try:
        print('answer_test', answer)
        if answer == 'NO':
            print('parse_answer not ok')
            return '0'
        elif answer == 'YES':
            print('parse_answer ok')
            return '1'
        else:
            print('parse_answer error')
            return "error"
    except json.JSONDecodeError:
        return "error"

def draw_random_cards_with_orientation(json_file_path, num_cards=3):
    """
    从指定的 JSON 文件中随机抽取指定数量的卡片，并随机赋予正逆位。
    
    参数:
    - json_file_path (str): JSON 文件的路径
    - num_cards (int): 要抽取的卡片数量

    返回:
    - random_cards_json (str): 随机抽取并赋予正逆位的卡片（JSON 格式）
    """
    # 从文件中加载卡片列表
    with open(json_file_path, 'r', encoding='utf-8') as f:
        tarot_cards = json.load(f)
    
    # 随机抽取卡片
    random_cards = random.sample(tarot_cards, num_cards)
    
    # 随机赋予正逆位
    orientations = ["正位", "逆位"]
    random_cards_with_orientation = [f"{card}（{random.choice(orientations)}）" for card in random_cards]
    
    # 转换为 JSON 格式
    random_cards_json = json.dumps(random_cards_with_orientation, ensure_ascii=False)
    
    return random_cards_json

def num_to_chinese(num_str: str) -> str:
    digits = {
        '0': '零',
        '1': '一',
        '2': '二',
        '3': '三',
        '4': '四',
        '5': '五',
        '6': '六',
        '7': '七',
        '8': '八',
        '9': '九'
    }
    units = ['', '十', '百', '千']
    
    if not num_str.isdigit():
        return num_str
    
    num_len = len(num_str)
    if num_len > 4:
        return num_str  # 如果数字超过4位，不转换

    result = ''
    zero_flag = False
    for idx, char in enumerate(num_str):
        if char == '0':
            zero_flag = True
        else:
            if zero_flag:
                result += digits['0']
                zero_flag = False
            result += digits[char] + units[num_len - idx - 1]
    return result

def transform_text(text: str) -> str:
    import re
    
    # 将逗号、句号、感叹号替换为 |
    text = text.replace('，', '|').replace('。', '|').replace('！', '|')
    
    # 移除括号和引号
    remove_chars = ['「', '」', '“', '”', '(', ')', '[', ']', '{', '}', '"', "'"]
    for char in remove_chars:
        text = text.replace(char, '')
    
    # 使用正则替换所有阿拉伯数字为中文数字
    text = re.sub(r'\d+', lambda m: num_to_chinese(m.group()), text)
    
    return text


def get_emotional_intent(user_input):
    """输入用户问题，返回用户意图，如果识别为非意图，则返回{"intend":"no"}

    Args:
        user_input (str): 用户输入的问题

    Returns:
        str: json 格式
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "请站在心理咨询师的角度，从用户的输入中识别用户的情感意图，并用 json 的格式回复；\n例如\n===\n\"input\":\"我明天运势怎么样？\"\n\"output\":{\"attend\":\"渴望获得好运，对现状可能不满\"}\n===\n如果不是心理咨询范畴的，或与主题无关的内容，请回复 {\"attend\":\"no\"}，\n答案只能是 json 格式的意图或 “no”，绝对不可以是其他的。\n===\n输入如下："
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['message']['content']


def get_tarot_response(question):
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "你是一个直播间的主播，现在直播的内容是塔罗占卜，但用户问了一个不属于塔罗可以解答的问题，请你根据他的问题，给出一段回答，目标是告诉用户，你只能回答塔罗相关的问题。\n\n\n在回答过程中，适当加入语气助词，增加一些人情味，答案尽量简短，高效。"
            },
            {
                "role": "user",
                "content": "{\"question\":\"" + question + "\"}"
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].message['content']

def get_final_answer(user_question):
    response = is_tarot_question(user_question)
    is_question = parse_answer(response)

    filename = 'utils\\danmu.txt'

    
    if is_question == '1':
    #     cards = draw_random_cards_with_orientation('./tarot_cards.json', num_cards=3)
    #     intend = get_emotional_intent(user_question)
    #     answer = tarot_answer(user_question, intend, cards)
    #     final_answer = transform_text(answer)
    #     return final_answer
    # else:
    #     answer = get_tarot_response(user_question)
    #     final_answer = transform_text(answer)
    #     return final_answer
        start_time = time.time()

        # 输出固定guide音频 
        
        AUDIO_DIR = 'wav\\guide_wav'
        TRACKING_FILE = 'selected_audios.txt'
        wav_data_PATH=get_random_audio(AUDIO_DIR,TRACKING_FILE)
        async_play_wav_windows(wav_data_PATH)

        cards = draw_random_cards_with_orientation('./tarot_cards.json', num_cards=3)
        intend = get_emotional_intent(user_question)
        answer = tarot_answer(user_question, intend, cards)
        final_answer = transform_text(answer)
        # return final_answer

        # answer,cards = tarot_answer(content)
        # final_answer = transform_text(answer)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Code executed in {execution_time:.2f} seconds")





        start_time = time.time()
        # time.sleep(5)
        
        user_question="关于你提到的"+user_question+"这个问题，下面我们进行抽卡占卜，请在心中默念你的问题。"
        wav_data_PATH=request_and_save_wav(user_question, "zh")

        async_play_wav_windows(wav_data_PATH)

        print("正在执行音频转换请求")
        wav_data_PATH = request_and_save_wav(final_answer, "zh")
        async_play_wav_windows(wav_data_PATH)

        # 更新前显的弹幕
        # 文件名
        # filename = 'utils\\danmu.txt'

        # 要写入的文本
        # text = nickname + "，您抽到的牌是：" + cards
        text = "您抽到的牌是：" + cards

        # 使用'with'语句确保文件正确关闭，并指定使用'utf-8'编码
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(text + '\n')  # '\n' 添加一个新行符

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Code executed in {execution_time:.2f} seconds")
        time.sleep(5)
        
    # elif is_question == "0":
        # print('NO')
        # 输出不能回答
        # print("问题无法回答")
        # async_play_wav_windows("refuse_answer_wav\0c2b3322-7545-11ee-80e9-0242ac110005.wav")

    else:
        print("问题无法回答")
        answer = get_tarot_response(user_question)
        final_answer = transform_text(answer)

        print("正在执行音频转换请求")
        wav_data_PATH = request_and_save_wav(final_answer, "zh")
        async_play_wav_windows(wav_data_PATH)

        # async_play_wav_windows("refuse_answer_wav\\0c2b3322-7545-11ee-80e9-0242ac110005.wav")
        return final_answer

        print('error')




    
# 调用函数，并传入用户的问题
if __name__ == '__main__':
        # user_question = "lonely"
        # response = is_tarot_question(user_question)
        # # print(response)
        # is_question = parse_answer(response)
        # # print('isquestion', is_question)
        # if is_question == '1':
        #     cards = draw_random_cards_with_orientation('./tarot_cards.json', num_cards=3)
        #     intend = get_emotional_intent(user_question)
        #     answer = tarot_answer(user_question, intend, cards)
        #     final_answer = transform_text(answer)
        #     print(final_answer)
        # else :
        #     answer = get_tarot_response(user_question)
        #     final_answer = transform_text(answer) 
        # 使用方法示例:
        user_input = "lonely"
        result = get_final_answer(user_input)
        print(result)



        

        



        
