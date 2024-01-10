import requests
import asyncio
import random
import tomli
import sys
import openai

# 读入配置
config_file = "config.toml"
if len(sys.argv) == 2:
  config_file = sys.argv[1]

with open(config_file, "rb") as f:
  config = tomli.load(f)

allowed_group = config["allowed"]["group"]
allowed_private = config["allowed"]["private"]
address = config["bind"]["address"]
port = config["bind"]["port"]
endpoint = config["cqhttp_api"]["endpoint"]
api_key = config["chat_gpt"]["key"]

print("----------READ FROM CONFIG FILE----------")
print("ALLOWED_GROUP:", allowed_group)
print("ALLOWED_PRIVATE:", allowed_private)
print("ADDRESS:", address)
print("PORT:", port)
print("ENDPOINT:", endpoint)
print("API_KEY", api_key)
print("-----------------------------------------")

"""替换掉消息内容中的特殊字符
Args:
  msg(str): 原来的消息字符串

Returns:
  str: 替换后的消息字符串
"""
def cleanup(msg):
  msg = str(msg)
  signs = { "&": "%26", "+": "%2B", "#": "%23" }
  for k, v in signs.items():
    msg = msg.replace(k, v)
  return msg
 
"""发送私聊消息

Args:
  uid(int): 目标QQ
  msg(str): 待发送的消息内容
  reply_msg_id(int): 这条消息所回复的对应消息的 ID
"""
def send_private_msg(uid, msg, reply_msg_id):
  msg = cleanup(msg)
  msg = "[CQ:reply,id={}]{}".format(reply_msg_id, msg)
  req_url = "{}/send_private_msg?&user_id={}&message={}".format(endpoint, uid, msg)
  requests.get(url=req_url)
 
"""发送群聊消息

Args:
  gid(int): 目标群号
  msg(str): 待发送的消息内容
  reply_msg_id(int): 这条消息所回复的对应消息的 ID
"""
def send_group_msg(gid, msg, reply_msg_id):
  msg = cleanup(msg)
  msg = "[CQ:reply,id={}]{}".format(reply_msg_id, msg)
  req_url = "{}/send_group_msg?&group_id={}&message={}".format(endpoint, gid, msg)
  requests.get(url=req_url)

"""
bots(dict):
  key(str): bid
  val(Chatbot)
"""
bots = {}

"""
对于一个 Bot 只有前一个问题回答完后，后一个问题才会被提交给 New bing
locks(dict):
  key(str): bid
  val(lock)

TODO: 应该使用同步队列代替单个锁，现在的实现只能保证两个问题的顺序
"""
locks = {}

"""向 New Bing 提问

Args:
  bid(str): 每个人独有的索引
  msg(str): 向 New Bing 提问的内容
Returns:
  str: New Bing 的回答
"""
async def ask(bid, msg):
  global bots
  global api_key
  bot = None
  answer = None

  try:
    if bid in bots:
      bots[bid].append({"role": "user", "content": msg})
    else:
      openai.api_key = random.choice(api_key)
      models = openai.Model.list()
      print(models)
      bot = [{"role": "user", "content": msg}]
      bots[bid] = bot
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=bots[bid])
    answer = chat_completion.choices[0].message.content
    bots[bid].append({"role": "assistant", "content": answer})

  except Exception as e:
    answer = str(e) + ": 向 ChatGPT 提问失败，可能是这个对话到达了 Token 上限。Bot 好像似了，但是又没似，你可以重试一下"
    if bid in bots:
      bots.pop(bid)

  return answer

"""与 Bot 私聊

Args:
  uid(int): 聊天者的 QQ
  msg(str): 聊天者发送的消息
  msg_id(int): 该条消息的 ID
"""
async def chat_private(uid, msg, msg_id):
  global bots
  global locks
  bid = str(uid)
  if bid not in locks:
    lock = asyncio.Lock()
    locks[bid] = lock
  lock = locks[bid]
  async with lock:
    answer = await ask(bid, msg)
    send_private_msg(uid, answer, msg_id)

"""在群里与 Bot 聊天

Args:
  gid(int): 群号
  uid(int): 聊天者的 QQ
  msg(str): 聊天者发送的消息
  msg_id(int): 该条消息的 ID
"""
async def chat_ingroup(gid, uid, msg, msg_id):
  global bots
  global locks

  # 以 "g " 开头的消息才被识别为与 Bot 聊天
  if msg[:2] != "g ":
    return
  msg = msg[2:]

  # 给群用户分配一个单独的 ID
  bid = str(gid) + '.' + str(uid)
  if bid not in locks:
    lock = asyncio.Lock()
    locks[bid] = lock
  lock = locks[bid]
  async with lock:
    answer = await ask(bid, msg)
    send_group_msg(gid, answer, msg_id)
