import os
import time
import json
import openai
import collections
from datetime import datetime
from plugins import register, Plugin, Event, logger, Reply, ReplyType
from bot.chatgpt import ChatGPTBot
from channel.message import Message


@register
class App(Plugin):
    name = 'wechat_summary'
    latest_stored = 0

    def __init__(self, config: dict):
        super().__init__(config)
        self.bot = ChatGPTBot()
        self.groups = self.store()

    def help(self, **kwargs):
        return '总结聊天记录'

    @property
    def commands(self):
        cmds = self.config.get('command', '总结聊天记录')
        if not isinstance(cmds, list):
            cmds = [cmds]
        return cmds

    def store(self, data=None):
        dir = os.path.dirname(os.path.abspath(__file__))
        path = f'{dir}/records.json'
        encoding = 'utf-8'
        try:
            if data is None:
                with open(path, mode='r', encoding=encoding) as file:
                    return json.load(file) or {}
            with open(path, mode='w', encoding=encoding) as file:
                json.dump(data, file, indent=1, ensure_ascii=False, cls=JsonEncoder)
                return data
        except Exception as exc:
            logger.warning('Store chat records fail: %s', exc)
            return {}

    def append_message(self, sender, msg):
        maxlen = self.config.get('maxlen', {})
        if isinstance(maxlen, dict):
            maxlen = maxlen.get(sender, maxlen.get('*', 200))
        if maxlen <= 0:
            return
        if sender not in self.groups:
            self.groups.setdefault(sender, [])
        if isinstance(self.groups[sender], list):
            self.groups[sender] = collections.deque(self.groups[sender], maxlen)
        if isinstance(msg, Message):
            msg = {
                'uid': msg.sender_id,
                'sender': msg.sender_name,
                'time': msg.create_time,
                'content': msg.content,
            }
        self.groups[sender].append(msg)

    def did_receive_message(self, event: Event):
        msg = event.message
        for cmd in self.commands:
            if f'{msg.content}'.endswith(cmd):
                return
        sender = msg.room_id or msg.sender_id
        self.append_message(sender, msg)
        now = time.time()
        if now - self.latest_stored > 10:
            self.store(self.groups)
        self.latest_stored = now

    def will_generate_reply(self, event: Event):
        query = event.context.query
        if query in self.commands:
            msg = event.message
            sender = msg.room_id or msg.sender_id
            event.reply = self.reply(sender)
            event.bypass()

    def will_decorate_reply(self, event: Event):
        pass

    def will_send_reply(self, event: Event):
        pass

    def reply(self, sender) -> Reply:
        lst = self.groups.get(sender) or []
        count = len(lst)
        minlen = self.config.get('minlen', {})
        if isinstance(minlen, dict):
            minlen = minlen.get(sender, minlen.get('*', 3))
        if count < minlen:
            return Reply(ReplyType.TEXT, '待总结的消息消息过少')

        tim = f'{datetime.now()}'
        prompt = self.config.get('prompt') or (
            '这是一份聊天记录，请总结成一份不超过500字的中文概述。'
            '总结正文里需要体现出这份聊天记录是从几点几分开始的。'
            '如果消息数量过多，可以挑选重点或者最新的记录来总结。'
            '遇到以下特征时可能是重点内容：'
            '- 多人复述的聊天内容'
            '- 多人回复 6/666/牛/牛逼 之前的内容'
        )
        prompt = f'当前时间为: {tim}\n{prompt}'
        session = [{"role": "system", "content": prompt}]

        lst = self.groups.pop(sender, None) or []
        count = len(lst)
        logger.info('Summarize %s messages for %s.', count, sender)

        if lst and lst[0].get('uid') == 'bot':
            pre = lst.pop(0) if isinstance(lst, list) else lst.popleft()
            session.append({"role": "assistant", "content": pre.get('content', '')})

        his = ['发言人,时间,内容']
        for msg in lst:
            row = ','.join([
                msg.get('sender', ''),
                msg.get('time', ''),
                msg.get('content', ''),
            ])
            his.append(row)
        session.append({"role": "user", "content": '\n'.join(his)})
        result = self.bot.reply_text(session)
        if msg := result.get('content'):
            self.append_message(sender, {
                'uid': 'bot',
                'sender': '前文总结',
                'time': tim,
                'content': msg,
            })
            logger.info('Summarized %s messages: %s.', count, result)
        if exc := result.get('exception'):
            if isinstance(exc, openai.error.APIError):
                msg = exc.user_message or f'{exc}'
        reply = Reply(ReplyType.TEXT, msg or '总结失败')
        return reply

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, collections.deque):
            return list(obj)
        return super().default(obj)