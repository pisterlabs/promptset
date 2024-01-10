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
from .token_calculator import filter_history, safe_token_count
from math import floor


APP_ID = ""
APP_SECRET = ""
VERIFICATION_TOKEN = ''


FIRST_TEXT = "这里是FeishuGPT，可以通过飞书机器人与ChatGPT进行对话。直接在这里发送您想要问ChatGPT的问题，然后您会被拉进一个群聊中与ChatGPT对话。请确保您发送的消息是纯文本，否则不会被回复。当前问题不会发送给ChatGPT，如果想要问问题请再发送一遍。"
HELP_TEXT = "这里是FeishuGPT，可以通过飞书机器人与ChatGPT进行对话。直接在这里发送您想要问ChatGPT的问题，然后您会被拉进一个群聊中与ChatGPT对话。请确保您发送的消息是纯文本，否则不会被回复。"
NORMAL_TEXT = "正在创建群聊，请稍等。\n在群聊中，如果想要保留上下文，请回复ChatGPT发送的信息。直接发送信息会被当成新的对话。"
SYSTEM_MSG = '''I am ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.
Knowledge cutoff: 2021-09
Current date: '''


all_event_id = []
chattings = {}  # 仅包含群聊，不包含和用户的私聊
users = {}
create_user_lock = threading.Lock()
token = ''
v2_created_time = -1


path = os.path.expanduser('~')
isWindows = ((platform.platform().lower().find('indows')) >= 0)
slash = {True: '\\', False: '/'}
SAVE_PATH = path + slash[isWindows] + ".jtc_feishu_gpt_3.5_mxTIPBpWbIxrLvZr5CCYpRwfY7DLrQRTxYlxxBWWrg3.txt"
V2_PATH = path + slash[isWindows] + ".jtc_feishu_gpt_3.5_mxTIPBpWbIxrLvZr5CCYpRwfY7DLrQRTxYlxxBWWrg3_v2.txt"


def get_tokens(text):
    enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    l = len(enc.encode(text))
    return l


def generate_card(answer, msg_id, chat_id):
    c = {
        "config": {
            "update_multi": True,
            "wide_screen_mode": True
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "content": answer,
                    "tag": "plain_text"
                }
            },
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {
                            "tag": "plain_text",
                            "content": "Regenerate"
                        },
                        "type": "primary",
                        "value": {
                            "action": "regenerate",
                            "msg_id": msg_id,
                            "chat_id": chat_id
                        }
                    }
                ]
            }
        ]
    }
    return json.dumps(c, ensure_ascii=False)


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
        'users': users_str,
        'v2_created_time': v2_created_time
    }
    all = json.dumps(all_dict, ensure_ascii=False)
    return all


def all_from_str(string):
    global all_event_id, chattings, users, v2_created_time
    all_dict = json.loads(string)
    all_event_id = all_dict['all_event_id']
    if ('v2_created_time' in all_dict):
        v2_created_time = all_dict['v2_created_time']
    else:
        v2_created_time = round(time() * 1000)
    for k, v in all_dict['chattings'].items():
        try:
            chattings[k] = Chat().from_str(v)
        except:
            chattings[k] = Chat_V2().from_str(v)
    for k, v in all_dict['users'].items():
        users[k] = User().from_str(v)


def save_all():
    s = all_to_str()
    f = open(V2_PATH, 'w')
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
    start_new_thread(check_regenerate, (msg,))
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
        parent_id = ''
        if ('parent_id' in msg['event']['message']):
            parent_id = msg['event']['message']['parent_id']
        if (str(type(chattings[chat_id])).find("_V2") >= 0):
            if (len(text) > 9 and text[0:9] == "@ChatGPT "):
                text = text[9:]
            chattings[chat_id].refresh_msg(text, msg_id, parent_id, int(msg['event']['message']['create_time']))
        else:
            chattings[chat_id].refresh_msg(text, msg_id)
        return
    save_all()


def check_regenerate(body: dict):
    if (body['action']['value']['action'] != 'regenerate'):
        return
    if (body["open_chat_id"] not in chattings):
        return
    chattings[body["open_chat_id"]].regenerate(body["open_message_id"])


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
        self.last_refreshed_time = 0
        self.resp_lock = threading.Lock()
        self.refresh_lock = threading.Lock()
        self.inited = False
        pass

    def create(self, open_id, chat_id):
        if (self.inited):
            raise Exception('Cannot initialize twice.')
        self.inited = True
        self.open_id = open_id
        self.chat_id = chat_id
        self.last_refreshed_time = round(time() * 1000)
        return self

    def to_str(self):
        d = {
            'chattings': self.chattings,
            'open_id': self.open_id,
            'chat_id': self.chat_id,
            'message_ids': self.message_ids,
            'last_refreshed_time': self.last_refreshed_time
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
        if ('last_refreshed_time' in d):
            self.last_refreshed_time = d['last_refreshed_time']
        else:
            self.last_refreshed_time = round(time() * 1000)
        return self

    def add_msg_id(self, msg_id):
        self.message_ids.append(msg_id)

    def refresh(self):
        with self.refresh_lock:
            t = floor(self.last_refreshed_time / 1000)
            header = {
                'Authorization': 'Bearer ' + token
            }
            url = f'https://open.feishu.cn/open-apis/im/v1/messages?container_id={self.chat_id}&container_id_type=chat&page_size=50&start_time={t}'
            r = None
            success = False
            while (success == False):
                r = http(url, Header=header)
                try:
                    if (r['text']['code'] == 0):
                        success = True
                        break
                except:
                    pass
                sleep(0.5)
            unrecorded = []
            for msg in r['text']['data']['items']:
                if (msg['sender']['sender_type'] != 'user'):
                    continue
                if (msg['message_id'] in self.message_ids):
                    continue
                if (msg['msg_type'] != 'text'):
                    continue
                id = msg['message_id']
                content = msg['body']['content']
                text = json.loads(content)['text']
                if ('mentions' in msg):
                    mentions = msg['mentions']
                    for m in mentions:
                        k = m['key']
                        n = m['name']
                        text = text.replace(k, f'@{n}')
                unrecorded.append((text, id))
            if (len(r['text']['data']['items']) >= 1):
                self.last_refreshed_time = int(r['text']['data']['items'][-1]['create_time']) - 1
            for u in unrecorded:
                start_new_thread(self.response, (u[1], u[0], False))

    def response(self, msg_id, text, first):
        with self.resp_lock:
            if (msg_id in self.message_ids):
                return
            self.add_msg_id(msg_id)
        start_new_thread(self.refresh, ())
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
        if (safe_token_count(text) > 3800):
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
            success = False
            for i in range(0, 2):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        temperature=2,
                        top_p=0,
                        max_tokens=min(2048, 4086 - safe_token_count(text) - safe_token_count(system_msg)),
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": text}
                        ]
                    )
                    answer1 = response.choices[0].message.content
                    success = True
                    break
                except:
                    sleep(1)
            if (success == True):
                answer = answer1
            else:
                answer = "系统信息：OpenAI API 出现异常"
            finished[1] = answer
            return answer

        def create_group_chat(finished):
            current_chat = Chat_V2().create(self.open_id, finished, text, system_msg)
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
            if (safe_token_count(msg) > 3800):
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
                model="gpt-3.5-turbo",
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


class Message:
    def __init__(self):
        self.text = ''
        self.type = ''  # "user" "gpt" "system"
        self.repliable = False
        self.parent = ''
        self.time = -1
        self.msg_id = ''
        self.inited = False

    def create(self, text, _type, repliable, parent, _time, id):
        if (self.inited):
            raise Exception('Cannot initialize twice.')
        self.inited = True
        self.text = text
        self.type = _type
        self.repliable = repliable
        self.parent = parent
        self.time = int(_time)
        self.msg_id = id
        return self

    def from_str(self, string):
        if (self.inited):
            raise Exception('Cannot initialize twice.')
        self.inited = True
        d = json.loads(string)
        self.text = d['text']
        self.type = d['type']
        self.repliable = d['repliable']
        self.parent = d['parent']
        self.time = d['time']
        self.msg_id = d['msg_id']
        return self

    def to_str(self):
        d = {
            'text': self.text,
            'type': self.type,
            'repliable': self.repliable,
            'parent': self.parent,
            'time': self.time,
            'msg_id': self.msg_id
        }
        s = json.dumps(d, ensure_ascii=False)
        return s


class Chat_V2:
    def __init__(self):
        self.chat_id = ''
        self.open_id = ''
        self.messages = {}
        self.inited = False
        self.refresh_lock = threading.Lock()  # 应该用不到，但先留着
        self.last_msg = ''
        self.system_msg = ''
        self.last_refreshed_time = 0

    def from_str(self, string):
        if (self.inited):
            raise Exception('Cannot initialize twice.')
        self.inited = True
        d = json.loads(string)
        self.chat_id = d['chat_id']
        self.open_id = d['open_id']
        self.last_msg = d['last_msg']
        self.system_msg = d['system_msg']
        self.last_refreshed_time = d['last_refreshed_time']
        for k, v in d['messages'].items():
            self.messages[k] = Message().from_str(v)
        return self

    def to_str(self):
        m = {}
        for k, v in self.messages.items():
            m[k] = v.to_str()
        d = {
            'chat_id': self.chat_id,
            'open_id': self.open_id,
            'last_msg': self.last_msg,
            'messages': m,
            'system_msg': self.system_msg,
            'last_refreshed_time': self.last_refreshed_time
        }
        s = json.dumps(d, ensure_ascii=False)
        return s

    def refresh(self):
        with self.refresh_lock:
            t = floor(self.last_refreshed_time / 1000)
            header = {
                'Authorization': 'Bearer ' + token
            }
            url = f'https://open.feishu.cn/open-apis/im/v1/messages?container_id={self.chat_id}&container_id_type=chat&page_size=50&start_time={t}'
            r = None
            success = False
            while (success == False):
                r = http(url, Header=header)
                try:
                    if (r['text']['code'] == 0):
                        success = True
                        break
                except:
                    pass
                sleep(0.5)
            unrecorded = []
            for msg in r['text']['data']['items']:
                if (msg['sender']['sender_type'] != 'user'):
                    continue
                if (msg['message_id'] in self.messages):
                    continue
                if (msg['msg_type'] != 'text'):
                    continue
                id = msg['message_id']
                create_time = int(msg['create_time'])
                parent_id = ''
                if ('parent_id' in msg):
                    parent_id = msg['parent_id']
                content = msg['body']['content']
                text = json.loads(content)['text']
                if ('mentions' in msg):
                    mentions = msg['mentions']
                    for m in mentions:
                        k = m['key']
                        n = m['name']
                        text = text.replace(k, f'@{n}')
                if (len(text) > 9 and text[0:9] == "@ChatGPT "):
                    text = text[9:]
                unrecorded.append((text, id, parent_id, create_time))
            if (len(r['text']['data']['items']) >= 1):
                self.last_refreshed_time = int(r['text']['data']['items'][-1]['create_time']) - 1
            for u in unrecorded:
                start_new_thread(self.refresh_msg, u)

    def regenerate(self, msg_id):
        if (msg_id not in self.messages):
            return
        if (self.messages[msg_id].repliable == False and self.messages[msg_id].text != "系统信息：OpenAI API 出现异常"):
            return
        start_new_thread(self.refresh, ())
        msg_id = self.messages[msg_id].parent
        history = self.generate_history(msg_id)
        print(history)
        messages, token_limit = filter_history(history)
        success = False
        for i in range(0, 2):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=2 - randint(0, 1000) / 1000,
                    top_p=0 + randint(0, 400) / 1000,
                    max_tokens=token_limit,
                    messages=messages
                )
                answer1 = response.choices[0].message.content
                success = True
                break
            except:
                sleep(1)
        if (success == True):
            answer = answer1
            msg_type = 'gpt'
            repliable = True
        else:
            answer = "系统信息：OpenAI API 出现异常"
            msg_type = 'system'
            repliable = False
        url = f'https://open.feishu.cn/open-apis/im/v1/messages/{msg_id}/reply'
        header = {
            "Content-Type": "application/json; charset=utf-8",
            'Authorization': 'Bearer ' + token
        }
        uuid = str(randint(10**39, 10**40 - 1))
        body = {
            "msg_type": "interactive",
            "uuid": uuid,
            "content": generate_card(answer, msg_id, self.chat_id)
        }
        body = json.dumps(body)
        success = False
        while (success == False):
            r = http(url, Method="POST", Header=header, Body=body)
            try:
                if (r['text']['code'] == 0):
                    answer_msg_id = r['text']['data']['message_id']
                    create_time = r['text']['data']["create_time"]
                    success = True
                    break
            except:
                pass
            sleep(0.5)
        self.messages[answer_msg_id] = Message().create(answer, msg_type, repliable, msg_id, int(create_time), answer_msg_id)
        self.last_msg = answer_msg_id
        save_all()

    def generate_history(self, msg_id):
        his = [{'role': 'system', "content": self.system_msg}]
        tmp = []
        current_id = msg_id
        while (True):
            current = self.messages[current_id]
            if (current.type == 'user'):
                tmp = [{'role': 'user', "content": current.text}] + tmp
            else:
                tmp = [{'role': 'assistant', "content": current.text}] + tmp
            if (current.parent == ""):
                break
            current_id = current.parent
        return his + tmp

    def refresh_msg(self, msg, msg_id, parent_id, c_time):
        if (msg_id in self.messages):
            return
        self.messages[msg_id] = Message().create(msg, 'user', False, parent_id, c_time, msg_id)
        start_new_thread(self.refresh, ())
        error_str = ''
        if (parent_id != "" and parent_id not in self.messages):
            # 回复 系统错误
            # 还有一种可能性，用户引用了gpt的回答，gpt的回答发送成功了但是系统未记录到，这种可能性忽略不计
            error_str = "系统错误"
        if (parent_id != "" and self.messages[parent_id].repliable == False and error_str == ""):
            # 回复 请勿回复您自己发送的/系统信息
            if (self.messages[parent_id].type == "user"):
                error_str = '请勿回复您自己发送的信息'
            else:
                error_str = '请勿回复系统信息'
        if (safe_token_count(msg) > 3800 and error_str == ""):
            # 长度超出限制
            error_str = "长度超出限制"
        # 以上三种情况，message type均为system
        if (error_str != ""):
            error_str = '系统信息：' + error_str
            url = f'https://open.feishu.cn/open-apis/im/v1/messages/{msg_id}/reply'
            header = {
                "Content-Type": "application/json; charset=utf-8",
                'Authorization': 'Bearer ' + token
            }
            uuid = str(randint(10**39, 10**40 - 1))
            body = {
                "msg_type": "text",
                "uuid": uuid,
                "content": json.dumps({'text': error_str}, ensure_ascii=False)
            }
            body = json.dumps(body)
            success = False
            while (success == False):
                r = http(url, Method="POST", Header=header, Body=body)
                try:
                    if (r['text']['code'] == 0):
                        answer_msg_id = r['text']['data']['message_id']
                        create_time = r['text']['data']["create_time"]
                        success = True
                        break
                except:
                    pass
                sleep(0.5)
            self.messages[answer_msg_id] = Message().create(error_str, 'system', False, msg_id, create_time, answer_msg_id)
            save_all()
            return
        save_all()
        history = self.generate_history(msg_id)
        messages, token_limit = filter_history(history)
        success = False
        for i in range(0, 2):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=2,
                    top_p=0,
                    max_tokens=token_limit,
                    messages=messages
                )
                answer1 = response.choices[0].message.content
                success = True
                break
            except:
                sleep(1)
        if (success == True):
            answer = answer1
            msg_type = 'gpt'
            repliable = True
        else:
            answer = "系统信息：OpenAI API 出现异常"
            msg_type = 'system'
            repliable = False
        url = f'https://open.feishu.cn/open-apis/im/v1/messages/{msg_id}/reply'
        header = {
            "Content-Type": "application/json; charset=utf-8",
            'Authorization': 'Bearer ' + token
        }
        uuid = str(randint(10**39, 10**40 - 1))
        body = {
            "msg_type": "interactive",
            "uuid": uuid,
            "content": generate_card(answer, msg_id, self.chat_id)
        }
        body = json.dumps(body)
        success = False
        while (success == False):
            r = http(url, Method="POST", Header=header, Body=body)
            try:
                if (r['text']['code'] == 0):
                    answer_msg_id = r['text']['data']['message_id']
                    create_time = r['text']['data']["create_time"]
                    success = True
                    break
            except:
                pass
            sleep(0.5)
        self.messages[answer_msg_id] = Message().create(answer, msg_type, repliable, msg_id, int(create_time), answer_msg_id)
        self.last_msg = answer_msg_id
        save_all()

    def create(self, open_id, finished, first_question, system_msg):
        if (self.inited):
            raise Exception('Cannot initialize twice.')
        self.inited = True
        self.open_id = open_id
        self.system_msg = system_msg
        self.last_refreshed_time = round(time() * 1000) - 1
        uuid = str(randint(10**39, 10**40 - 1))
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

        def todo():
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
                        question_msg_id = r['text']['data']['message_id']
                        create_time = r['text']['data']["create_time"]
                        success = True
                        break
                except:
                    pass
                sleep(0.5)
            self.messages[question_msg_id] = Message().create(first_question, 'user', False, '', int(create_time), question_msg_id)
            while (True):
                if (finished[1] != False):
                    break
            answer = finished[1]
            if (answer == "系统信息：OpenAI API 出现异常"):
                msg_type = "system"
                repliable = False
            else:
                msg_type = 'gpt'
                repliable = True
            url = f'https://open.feishu.cn/open-apis/im/v1/messages/{question_msg_id}/reply'
            header = {
                "Content-Type": "application/json; charset=utf-8",
                'Authorization': 'Bearer ' + token
            }
            uuid = str(randint(10**39, 10**40 - 1))
            body = {
                "msg_type": "interactive",
                "uuid": uuid,
                "content": generate_card(answer, question_msg_id, self.chat_id)
            }
            body = json.dumps(body)
            success = False
            while (success == False):
                r = http(url, Method="POST", Header=header, Body=body)
                try:
                    if (r['text']['code'] == 0):
                        answer_msg_id = r['text']['data']['message_id']
                        create_time = r['text']['data']["create_time"]
                        success = True
                        break
                except:
                    pass
                sleep(0.5)
            self.messages[answer_msg_id] = Message().create(answer, msg_type, repliable, question_msg_id, int(create_time), answer_msg_id)
            self.last_msg = answer_msg_id
            save_all()
        start_new_thread(todo, ())
        return self


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
        try:
            f = open(V2_PATH, 'r')
        except:
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
