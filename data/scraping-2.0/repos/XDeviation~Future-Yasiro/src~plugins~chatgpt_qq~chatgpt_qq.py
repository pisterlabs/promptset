import os
import openai
import json
import pickle
import threading
import time
# import Future Yasiro improve package
import sys
# sys.path.append(r'd:\ChatGPT\Future-Yasiro\Future-Yasiro\src\plugins\chatgpt_qq')
sys.path.append('../../')
from logger import Logger
from mq_utils import get_mq_connection
from config import CHATGPT_API_KEY, DATA_FILE, TOKEN_LIMIT, CONTEXT_TIME_LIMIT

# You can find your log in src/logs/{log_file}
logger = Logger(__name__, log_file="chatgpt_qq.log")

openai.api_key = CHATGPT_API_KEY
user_list = []
message_cache = ""

# receive_message(user_info, message) <--  receive_message_api()
#  ↓
# send_message(user_info, context)     --> send_message_api()

# user_list中存储的对话数据的结构
# user_list = [
#     conversation1,
#     conversation2,
#     conversation3 = {
#         "user_info": user_info = {},
#         "context": context = [
#             context1,
#             context2,
#             context3 = {
#                 "role": "user",
#                 "content": message
#             },
#             ...
#         ],
#         "tokens": string2token(all_messages),
#         "last_time": last_time
#     },
#     ...
# ]
# 粗略估计，2 token = 8 char = 1 汉字（略微高估防止超过限制）
# 实际计算方式见 https://platform.openai.com/tokenizer
def string2token(s):
    return len(s)*1.2

def init():
    # 重启bot时，读取聊天记录
    global user_list
    try:
        with open(DATA_FILE, 'rb') as f:
            user_list = pickle.load(f)
    except FileNotFoundError:
        with open(DATA_FILE, 'w'):
            pass
        user_list = []
    return

def add_new_user(user_info, message):
    # 一个新用户发来一条消息
    # context = [{"role": "user", "content": message}]
    context = []
    conversation = {
        "user_info": user_info,
        "context": context,
        "tokens": 0,    # string2token(message),
        "last_time": None
    }
    user_list.append(conversation)
    return conversation

def pop_append(conversation, message):
    # 在append新message进上下文时，要保证总长度不超过token_limit
    conversation["context"].append(message)
    tokens = conversation["tokens"]
    tokens += string2token(message["content"])
    while tokens >= TOKEN_LIMIT:
        tokens -= string2token(conversation["context"][0]["content"])
        conversation["context"].pop(0)
    conversation["tokens"] = tokens

def send_message(user_info, context):
    # 向用户发送消息
    global message_cache
    message_cache = context["content"]
    return

def receive_message(user_info, message):
    # 收到消息后的处理
    # 找到对话，若不存在则创建
    in_list = False
    conversation = {}
    for conv in user_list:
        if user_info["sender"]["nickname"] == "群成员":
            if conv["user_info"]["sender"]["nickname"] == "群成员":
                logger.info({
                    "action":"comparing two group",
                    "A":user_info,
                    "B":conv["user_info"],
                    "result":user_info == conv["user_info"]
                })
        if user_info == conv["user_info"]:
            in_list = True
            conversation = conv
            break
    if in_list == False:
        conversation = add_new_user(user_info, message)
    # 当距离上次对话过久，清除当前对话的上下文，以节省token资源
    last_time = conversation.get("last_time")
    cur_time = time.time()
    clean_flag = 1
    if last_time != None:
        if cur_time - last_time < CONTEXT_TIME_LIMIT:
            clean_flag = 0
    if clean_flag:
        conversation["context"] = []
        conversation["tokens"] = 0
    conversation["last_time"] = cur_time
    # 维护上下文，根据收到的信息进行回复
    pop_append(conversation, {"role": "user", "content": message})
    completion = None
    try_times = 3
    while completion == None:
        if try_times == 0:
            break
        try_times -= 1
        try:
            # 调用外部包的代码
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation["context"]
            )
        except Exception as e:
            # 处理错误的代码，例如记录日志、发送通知等
            logger.error({
                "error_description": "openai server error",
                "error_info":{
                    "res_try_times": try_times,
                    "exception_message": e
                }
            })
            completion = None
    if completion == None:
        pop_append(conversation, {"role": "assistant", "content":"ChatGPT has disconnected."})
        send_message(user_info, {"role": "assistant", "content":"ChatGPT has disconnected."})
    else:
        pop_append(conversation, completion.choices[0].message)
        send_message(user_info, completion.choices[0].message)
    # 更新聊天记录文件
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(user_list, f)
    return

# # 下面是一个无限循环与ChatGPT对话的示例代码
# GlobalMessages = []
# while True:
#     text = input()
#     GlobalMessages.append({"role": "user", "content": text})
#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=GlobalMessages
#     )
#     # print(json.dumps(completion.choices[0].message, ensure_ascii=False))
#     print(completion.choices[0].message["content"])
#     GlobalMessages.append(completion.choices[0].message)




def run_personal():
    mq_connection = get_mq_connection()
    mq_channel = mq_connection.channel()
    mq_channel.queue_declare(queue=".gpt")

    def callback(ch, method, properties, body):
        message = json.loads(body)
        message_type = message.get("message_type", "")
        if message_type != 'private' and message_type != 'group':
            return
        # if message_type == 'group':
        #     mes = message.get()
        #     if 1:
        #         return
        # 信息格式为".gpt message"，应当去掉前四个字符
        message_text = message.get("message")[4:]
        user_info = {
            "group": message_type == 'group',
            "message_type": message_type,
            "sender": message.get("sender")
        }
        if user_info["group"] == True:
            user_info["group_id"] = message.get("group_id")
            
        logger.info({
            "action": "receive",
            "data": {
                "user_info": user_info,
                "message_text": message_text
            }
        })

        receive_message(user_info, message_text)
            
        logger.info({
            "action": "send",
            "data": {
                "user_info": user_info,
                "message_text": message_cache
            }
        })

        send_mq_connection = get_mq_connection()
        send_mq_channal = send_mq_connection.channel()
        send_mq_channal.queue_declare(queue="send_message")
        return_msg = {
            "message_type": "qq_message",
            "message": None,
        }
        if message_type == 'private':
            user_id = message.get("user_id", 0)
            return_msg["message"] = { 
                "action": "send_private_msg",
                "params": {
                    "user_id": user_id, 
                    "message": message_cache,
                    "auto_escape": True
                }
            }
            

        if message_type == 'group':
            group_id = message.get("group_id", 0)
            return_msg["message"] = { 
                "action": "send_group_msg",
                "params": {
                    "group_id": group_id, 
                    "message": message_cache,
                    "auto_escape": True
                }
            }

        send_mq_channal.basic_publish(
            exchange='',
            routing_key="send_message",
            body=json.dumps(return_msg)
        )
        # logger.info(f"Send message {return_msg}")

    mq_channel.basic_consume(
        queue=".gpt", on_message_callback=callback, auto_ack=True
    )
    mq_channel.start_consuming()

    mq_connection.close()
    pass

def run_group():
    mq_connection_group = get_mq_connection()
    mq_channel_group = mq_connection_group.channel()
    mq_channel_group.queue_declare(queue=".gptg")

    def callback_group(ch, method, properties, body):
        message = json.loads(body)
        message_type = message.get("message_type", "")
        if message_type != 'group':
            return
        
        fake_group_sender = {
            'age': 0, 
            'nickname': '群成员', 
            'sex': 'unknown', 
            'user_id': message.get("group_id", 0)
        }
        # 信息格式为".gptg message"，应当去掉前五个字符
        message_text = message.get("message")[5:]
        user_info = {
            "group": message_type == 'group',
            "message_type": message_type,
            "sender": fake_group_sender
        }
        if user_info["group"] == True:
            user_info["group_id"] = message.get("group_id")
            
        logger.info({
            "action": "receive",
            "data": {
                "user_info": user_info,
                "message_text": message_text
            }
        })

        logger.info({
            "fake user_info: ": user_info,
        })

        receive_message(user_info, message_text)
            
        logger.info({
            "action": "send",
            "data": {
                "user_info": user_info,
                "message_text": message_cache
            }
        })

        send_mq_connection = get_mq_connection()
        send_mq_channal = send_mq_connection.channel()
        send_mq_channal.queue_declare(queue="send_message")
        return_msg = {
            "message_type": "qq_message",
            "message": None,
        }
        if message_type == 'private':
            user_id = message.get("user_id", 0)
            return_msg["message"] = { 
                "action": "send_private_msg",
                "params": {
                    "user_id": user_id, 
                    "message": message_cache,
                    "auto_escape": True
                }
            }
            

        if message_type == 'group':
            group_id = message.get("group_id", 0)
            return_msg["message"] = { 
                "action": "send_group_msg",
                "params": {
                    "group_id": group_id, 
                    "message": message_cache,
                    "auto_escape": True
                }
            }

        send_mq_channal.basic_publish(
            exchange='',
            routing_key="send_message",
            body=json.dumps(return_msg)
        )
        # logger.info(f"Send message {return_msg}")

    mq_channel_group.basic_consume(
        queue=".gptg", on_message_callback=callback_group, auto_ack=True
    )
    mq_channel_group.start_consuming()

    mq_connection_group.close()
    pass

def main():
    init()
    # 创建两个新线程
    thread1 = threading.Thread(target=run_personal)
    thread2 = threading.Thread(target=run_group)

    # 启动线程并等待线程结束
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

# 下面是一段测试receive_message的代码
# test_user_info = {
#     "group": True,
#     "message_type": "group",
#     "group_id": "35678",
#     "sender": {
#         "nickname": "zawedx",
#         "user_id": 1071022348
#     }
# }
# test_message_text = "你好！"
# receive_message(test_user_info, test_message_text)
# print(message_cache)