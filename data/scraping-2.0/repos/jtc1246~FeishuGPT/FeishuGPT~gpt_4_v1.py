from myHttp import http
import openai
import json
from time import time, sleep, gmtime
from _thread import start_new_thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from random import randint
import tiktoken
import os
import platform
from .token_calculator_8192 import filter_history, safe_token_count


APP_ID = ""
APP_SECRET = ""
VERIFICATION_TOKEN = ''


FIRST_TEXT = "这里是FeishuGPT，可以通过飞书机器人与ChatGPT进行对话，当前版本为GPT-4。直接在这里发送您想要问ChatGPT的问题，然后您会被拉进一个群聊中与ChatGPT对话。请确保您发送的消息是纯文本，否则不会被回复。GPT-4 API价格较贵，请合理控制使用量。当前问题不会发送给ChatGPT，如果想要问问题请再发送一遍。"
HELP_TEXT = "这里是FeishuGPT，可以通过飞书机器人与ChatGPT进行对话，当前版本为GPT-4。直接在这里发送您想要问ChatGPT的问题，然后您会被拉进一个群聊中与ChatGPT对话。请确保您发送的消息是纯文本，否则不会被回复。GPT-4 API价格较贵，请合理控制使用量。"
NORMAL_TEXT = "正在创建群聊，请稍等。"
SYSTEM_MSG = '''You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
Knowledge cutoff: 2021-09
Current date: '''


all_event_id = []
chattings = {}  # 仅包含群聊，不包含和用户的私聊
users = {}
create_user_lock = threading.Lock()
token = ''


path = os.path.expanduser('~')
isWindows = ((platform.platform().lower().find('indows')) >= 0)
slash = {True: '\\', False: '/'}
SAVE_PATH = path + slash[isWindows] + ".jtc_feishu_gpt_4_mxTIPBpWbIxrLvZr5CCYpRwfY7DLrQRTxYlxxBWWrg3.txt"


def get_tokens(text):
    enc = tiktoken.encoding_for_model('gpt-4')
    l = len(enc.encode(text))
    return l


def all_to_str():
    chattings_str = {}
    for k, v in chattings.items():
        if (v.inited == False):
            continue
        chattings_str[k] = v.to_str()
    users_str = {}
    for k, v in users.items():
        if (v.inited == False):
            continue
        users_str[k] = v.to_str()
    all_dict = {
        'all_event_id': all_event_id,
        'chattings': chattings_str,
        'users': users_str
    }
    all = json.dumps(all_dict, ensure_ascii=False)
    return all


def all_from_str(string):
    global all_event_id, chattings, users
    all_dict = json.loads(string)
    all_event_id = all_dict['all_event_id']
    for k, v in all_dict['chattings'].items():
        chattings[k] = Chat().from_str(v)
    for k, v in all_dict['users'].items():
        users[k] = User().from_str(v)


def save_all():
    s = all_to_str()
    f = open(SAVE_PATH, 'w')
    f.write(s)
    f.close()


class Resquest(BaseHTTPRequestHandler):
    def do_GET(self):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers['content-length']))
        body = body.decode('utf-8')
        print(body)
        body = json.loads(body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'')
        start_new_thread(process_msg, (body,))


def process_msg(msg: dict):
    if (msg['header']['token'] != VERIFICATION_TOKEN or msg['header']['app_id'] != APP_ID):
        return
    if (not (msg['header']['event_type'] == "im.message.receive_v1" and (msg['header']['event_id'] not in all_event_id))):
        return
    all_event_id.append(msg['header']['event_id'])
    if (msg['event']['message']['chat_type'] == 'p2p' and msg['event']['message']['message_type'] == 'text'):
        open_id = msg['event']['sender']['sender_id']['open_id']
        chat_id = msg['event']['message']['chat_id']
        msg_id = msg['event']['message']['message_id']
        content = msg['event']['message']['content']
        text = json.loads(content)['text']
        if ('mentions' in msg['event']['message']):
            mentions = msg['event']['message']['mentions']
            for m in mentions:
                k = m['key']
                n = m['name']
                text = text.replace(k, f'@{n}')
        first = False
        with create_user_lock:
            if (open_id not in users):
                user = User().create(open_id, chat_id)
                users[open_id] = user
                first = True
            else:
                user = users[open_id]
        user.response(msg_id, text, first)
        return
    if (msg['event']['message']['chat_type'] == 'group' and msg['event']['message']['message_type'] == 'text'):
        chat_id = msg['event']['message']['chat_id']
        msg_id = msg['event']['message']['message_id']
        if (chat_id not in chattings):
            return
        content = msg['event']['message']['content']
        text = json.loads(content)['text']
        if ('mentions' in msg['event']['message']):
            mentions = msg['event']['message']['mentions']
            for m in mentions:
                k = m['key']
                n = m['name']
                text = text.replace(k, f'@{n}')
        chattings[chat_id].refresh_msg(text, msg_id)
        return
    save_all()


def get_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    header = {"Content-Type": "application/json; charset=utf-8"}
    body = {
        "app_id": APP_ID,
        "app_secret": APP_SECRET
    }
    body = json.dumps(body).encode('utf-8')
    r = http(url, "POST", Header=header, Body=body)
    try:
        return r['text']['tenant_access_token']
    except:
        return ''


def update_token():
    global token
    while True:
        sleep(0.5)
        t = get_token()
        if (t != ''):
            token = t
            sleep(600)
            continue
        while (t == ''):
            t = get_token()
            sleep(0.5)
        token = t
        sleep(600)


class User:
    def __init__(self):
        self.chattings = []  # 群聊的chat_id
        self.open_id = ''
        self.chat_id = ''
        self.message_ids = []
        self.resp_lock = threading.Lock()
        self.inited = False
        pass

    def create(self, open_id, chat_id):
        if (self.inited):
            raise Exception('Cannot initialize twice.')
        self.inited = True
        self.open_id = open_id
        self.chat_id = chat_id
        return self

    def to_str(self):
        d = {
            'chattings': self.chattings,
            'open_id': self.open_id,
            'chat_id': self.chat_id,
            'message_ids': self.message_ids
        }
        s = json.dumps(d, ensure_ascii=False)
        return s

    def from_str(self, string):
        if (self.inited):
            raise Exception('Cannot initialize twice.')
        self.inited = True
        d = json.loads(string)
        self.chattings = d['chattings']
        self.open_id = d['open_id']
        self.chat_id = d['chat_id']
        self.message_ids = d['message_ids']
        return self

    def add_msg_id(self, msg_id):
        self.message_ids.append(msg_id)

    def refresh():
        pass

    def response(self, msg_id, text, first):
        with self.resp_lock:
            if (msg_id in self.message_ids):
                return
            self.add_msg_id(msg_id)
        if (first):
            url = f'https://open.feishu.cn/open-apis/im/v1/messages/{msg_id}/reply'
            header = {
                "Content-Type": "application/json; charset=utf-8",
                'Authorization': 'Bearer ' + token
            }
            uuid = str(randint(10**39, 10**40 - 1))
            body = {
                "msg_type": "text",
                "uuid": uuid,
                "content": json.dumps({'text': FIRST_TEXT}, ensure_ascii=False)
            }
            body = json.dumps(body)
            success = False
            while (success == False):
                r = http(url, Method="POST", Header=header, Body=body)
                try:
                    if (r['text']['code'] == 0):
                        success = True
                        break
                except:
                    pass
                sleep(0.5)
            save_all()
            return
        if (text == "使用说明"):
            url = f'https://open.feishu.cn/open-apis/im/v1/messages/{msg_id}/reply'
            header = {
                "Content-Type": "application/json; charset=utf-8",
                'Authorization': 'Bearer ' + token
            }
            uuid = str(randint(10**39, 10**40 - 1))
            body = {
                "msg_type": "text",
                "uuid": uuid,
                "content": json.dumps({'text': HELP_TEXT}, ensure_ascii=False)
            }
            body = json.dumps(body)
            success = False
            while (success == False):
                r = http(url, Method="POST", Header=header, Body=body)
                try:
                    if (r['text']['code'] == 0):
                        success = True
                        break
                except:
                    pass
                sleep(0.5)
            save_all()
            return
        if (safe_token_count(text) > 7800):
            url = f'https://open.feishu.cn/open-apis/im/v1/messages/{msg_id}/reply'
            header = {
                "Content-Type": "application/json; charset=utf-8",
                'Authorization': 'Bearer ' + token
            }
            uuid = str(randint(10**39, 10**40 - 1))
            body = {
                "msg_type": "text",
                "uuid": uuid,
                "content": json.dumps({'text': '系统信息：长度超出限制'}, ensure_ascii=False)
            }
            body = json.dumps(body)
            success = False
            while (success == False):
                r = http(url, Method="POST", Header=header, Body=body)
                try:
                    if (r['text']['code'] == 0):
                        success = True
                        break
                except:
                    pass
                sleep(0.5)
            save_all()
            return
        # 有建群、询问openai、私聊回复三件事，三件事同时做

        finished = [False, False, False, '']
        gmt = gmtime()
        y = gmt.tm_year
        m = str(gmt.tm_mon)
        d = str(gmt.tm_mday)
        m = '0' * (2 - len(m)) + m
        d = '0' * (2 - len(d)) + d
        system_msg = SYSTEM_MSG + f'{y}-{m}-{d}'

        def reply(finished):
            url = f'https://open.feishu.cn/open-apis/im/v1/messages/{msg_id}/reply'
            header = {
                "Content-Type": "application/json; charset=utf-8",
                'Authorization': 'Bearer ' + token
            }
            uuid = str(randint(10**39, 10**40 - 1))
            body = {
                "msg_type": "text",
                "uuid": uuid,
                "content": json.dumps({'text': NORMAL_TEXT}, ensure_ascii=False)
            }
            body = json.dumps(body)
            success = False
            while (success == False):
                r = http(url, Method="POST", Header=header, Body=body)
                try:
                    if (r['text']['code'] == 0):
                        success = True
                        break
                except:
                    pass
                sleep(0.5)
            finished[0] = True
            return

        def ask_openai(finished):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=2,
                top_p=0,
                max_tokens=min(2048, 8182 - safe_token_count(text) - safe_token_count(system_msg)),
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": text}
                ]
            )
            finished[1] = response.choices[0].message.content
            return response.choices[0].message.content

        def create_group_chat(finished):
            current_chat = Chat().create(self.open_id, finished, text, system_msg)
            chat_id = current_chat.chat_id
            self.chattings.append(chat_id)
            chattings[chat_id] = current_chat
            save_all()

        start_new_thread(reply, (finished,))
        start_new_thread(ask_openai, (finished,))
        start_new_thread(create_group_chat, (finished,))


class Chat:
    def __init__(self):
        self.chat_id = ''
        self.open_id = ''
        self.msgs = {}
        self.inited = False
        self.refresh_lock = threading.Lock()  # 暂时先这样做
        self.history = []  # 暂时先这样做

    def from_str(self, string):
        if (self.inited):
            raise Exception('Cannot initialize twice.')
        self.inited = True
        d = json.loads(string)
        self.chat_id = d['chat_id']
        self.open_id = d['open_id']
        self.msgs = d['msgs']
        self.history = d['history']
        return self

    def to_str(self):
        d = {
            'chat_id': self.chat_id,
            'open_id': self.open_id,
            'msgs': self.msgs,
            'history': self.history
        }
        s = json.dumps(d, ensure_ascii=False)
        return s

    def create(self, open_id, finished, first_question, system_msg):
        if (self.inited):
            raise Exception('Cannot initialize twice.')
        self.inited = True
        self.open_id = open_id
        uuid = str(randint(10**39, 10**40 - 1))
        self.history = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": first_question}
        ]
        url = f'https://open.feishu.cn/open-apis/im/v1/chats?user_id_type=open_id&uuid={uuid}'
        header = {
            "Content-Type": "application/json; charset=utf-8",
            'Authorization': 'Bearer ' + token
        }
        body = {
            "name": "New Chat",
            "user_id_list": [
                self.open_id
            ]
        }
        body = json.dumps(body)
        success = False
        while (success == False):
            r = http(url, Method="POST", Header=header, Body=body)
            try:
                if (r['text']['code'] == 0):
                    success = True
                    self.chat_id = r['text']['data']['chat_id']
                    break
            except:
                pass
            sleep(0.5)
        # 到此处群聊创建成功
        url = 'https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id'
        header = {
            "Content-Type": "application/json; charset=utf-8",
            'Authorization': 'Bearer ' + token
        }
        uuid = str(randint(10**39, 10**40 - 1))
        body = {
            'receive_id': self.chat_id,
            "msg_type": 'text',
            'content': json.dumps({'text': f"当前问题：\n{first_question}"}),
            'uuid': uuid
        }
        body = json.dumps(body)
        success = False
        while (success == False):
            r = http(url, Method="POST", Header=header, Body=body)
            try:
                if (r['text']['code'] == 0):
                    success = True
                    question_msg_id = r['text']['data']['message_id']
                    break
            except:
                pass
            sleep(0.5)
        while (True):
            if (finished[1] != False):
                break
        answer = finished[1]
        self.history.append({"role": "assistant", "content": answer})
        url = f'https://open.feishu.cn/open-apis/im/v1/messages/{question_msg_id}/reply'
        header = {
            "Content-Type": "application/json; charset=utf-8",
            'Authorization': 'Bearer ' + token
        }
        uuid = str(randint(10**39, 10**40 - 1))
        body = {
            "msg_type": "text",
            "uuid": uuid,
            "content": json.dumps({'text': answer}, ensure_ascii=False)
        }
        body = json.dumps(body)
        success = False
        while (success == False):
            r = http(url, Method="POST", Header=header, Body=body)
            try:
                if (r['text']['code'] == 0):
                    success = True
                    break
            except:
                pass
            sleep(0.5)
        return self

    def refresh(self):
        pass

    def refresh_msg(self, msg, msg_id):
        with self.refresh_lock:
            if (safe_token_count(msg) > 7800):
                url = f'https://open.feishu.cn/open-apis/im/v1/messages/{msg_id}/reply'
                header = {
                    "Content-Type": "application/json; charset=utf-8",
                    'Authorization': 'Bearer ' + token
                }
                uuid = str(randint(10**39, 10**40 - 1))
                body = {
                    "msg_type": "text",
                    "uuid": uuid,
                    "content": json.dumps({'text': '系统信息：长度超出限制'}, ensure_ascii=False)
                }
                body = json.dumps(body)
                success = False
                while (success == False):
                    r = http(url, Method="POST", Header=header, Body=body)
                    try:
                        if (r['text']['code'] == 0):
                            success = True
                            break
                    except:
                        pass
                    sleep(0.5)
                return
            self.history.append({"role": "user", "content": msg})
            messages, token_limit = filter_history(self.history)
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=2,
                top_p=0,
                max_tokens=token_limit,
                messages=messages
            )
            answer = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": answer})
            url = f'https://open.feishu.cn/open-apis/im/v1/messages/{msg_id}/reply'
            header = {
                "Content-Type": "application/json; charset=utf-8",
                'Authorization': 'Bearer ' + token
            }
            uuid = str(randint(10**39, 10**40 - 1))
            body = {
                "msg_type": "text",
                "uuid": uuid,
                "content": json.dumps({'text': answer}, ensure_ascii=False)
            }
            body = json.dumps(body)
            success = False
            while (success == False):
                r = http(url, Method="POST", Header=header, Body=body)
                try:
                    if (r['text']['code'] == 0):
                        success = True
                        break
                except:
                    pass
                sleep(0.5)
            save_all()


def get_msg(chat_id, start=0):
    # 私聊已测试没问题，群聊尚未测试
    url = f'https://open.feishu.cn/open-apis/im/v1/messages?container_id_type=chat&container_id={chat_id}&page_size=50'
    header = {
        "Content-Type": "application/json; charset=utf-8",
        'Authorization': 'Bearer ' + token
    }
    r = http(url, Header=header)
    try:
        items = r['text']['data']['items']
        result = []
        for item in items:
            if (item["msg_type"] != 'text'):
                continue
            # time sender message_id text parent_id
            time = item["create_time"]
            msg_id = item["message_id"]
            content = item['body']['content']
            sender = item['sender']['sender_type']
            text = json.loads(content)['text']
            if ('mentions' in item):
                mentions = item['mentions']
                for m in mentions:
                    k = m['key']
                    n = m['name']
                    text = text.replace(k, f'@{n}')
            if (sender != 'user'):
                sender = 'bot'
            parent_id = ''
            if ('parent_id' in item):
                parent_id = item['parent_id']
            result.append([
                time, sender, msg_id, text, parent_id
            ])
        return result
    except:
        return []


# if __name__ == '__main__':
#     start_new_thread(update_token, ())
#     server = ThreadingHTTPServer(('0.0.0.0', 80), Resquest)
#     start_new_thread(server.serve_forever, ())
#     try:
#         f = open(SAVE_PATH, 'r')
#         all_from_str(f.read())
#         f.close()
#         print("从本地文件加载数据成功")
#     except:
#         print("未找到本地文件或本地文件存在错误，无法访问历史数据（如果是第一次使用，请忽略此问题）")
#     sleep(2)
#     print(token)
#     while True:
#         sleep(10)


def start_async(feishu_app_id, app_secret, verification_token, openai_api_key, port):
    global APP_ID, APP_SECRET, VERIFICATION_TOKEN
    openai.api_key = openai_api_key
    APP_SECRET = app_secret
    APP_ID = feishu_app_id
    VERIFICATION_TOKEN = verification_token
    start_new_thread(update_token, ())
    server = ThreadingHTTPServer(('0.0.0.0', port), Resquest)
    start_new_thread(server.serve_forever, ())
    try:
        f = open(SAVE_PATH, 'r')
        all_from_str(f.read())
        f.close()
        print("从本地文件加载数据成功")
    except:
        print("未找到本地文件或本地文件存在错误，无法访问历史数据（如果是第一次使用，请忽略此问题）")
    sleep(2)
    if (token == ''):
        print('请检查app_id和app_secret是否正确')
    else:
        print('启动服务成功')
