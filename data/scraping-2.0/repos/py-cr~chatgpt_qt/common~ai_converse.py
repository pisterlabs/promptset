# -*- coding:utf-8 -*-
# title           :ai_converse.py
# description     :两AI聊天类
# author          :Python超人
# date            :2023-6-13
# link            :https://gitcode.net/pythoncr/
# python_version  :3.8
# ==============================================================================
from common.openai_chatbot import OpenAiChatbot
import time
import datetime

from db.db_ops import SessionOp, HistoryOp
from db.entities import History


class AiConverse:
    """
    两AI聊天类
    """

    def __init__(self, ai1_name="", ai2_name="", ai1_role="", ai2_role="", maximum_context_length=3000):
        """

        :param ai1_name: AI1名称
        :param ai2_name: AI2名称
        :param ai1_role: AI1角色
        :param ai2_role: AI2角色
        :param maximum_context_length: 最大上下文文字长度
        """
        self.ai2_role = ai2_role
        self.ai1_role = ai1_role
        self.ai2_name = ai2_name
        self.ai1_name = ai1_name
        self.maximum_context_length = maximum_context_length

        self.chatbot = OpenAiChatbot()
        # 聊天停止标记（默认为False）
        self.chat_stop = False

    def current_time(self):
        """
        获取当前的时间
        :return:
        """
        time_str = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
        return time_str

    def chat(self, messages):
        """
        聊天
        :param messages:
        :return:
        """
        if self.chat_stop:
            # 如果聊天终止则返回 1
            return 1, ""
        content = ""
        print(messages)
        for reply, status, is_error in self.chatbot.chat_messages(messages, None):
            reply_content = reply["content"]
            content += reply_content

            if callable(self.callback):
                # 如果有回调函数（用于传递状态）
                # "action": "message" 传递消息内容
                self.callback({"action": "message", "params": {"content": content, "part_content": reply_content}})
                if self.chat_stop:
                    # 如果聊天终止则返回 1
                    return 1, content

            if is_error != 0:
                if "maximum context length" in content:
                    self.history_messages_resize()
                    return 1, content
                elif "Rate limit reached for" in content:
                    if self.chat_stop:
                        # 如果聊天终止则返回 1
                        return 1, content
                    # time.sleep(20)
                    self.wait_sleep_sec(self.sleep_sec())
                    return 1, content
                else:
                    print("ERROR:", content)
                    # exit(-1)
                    return 1, content

        return 0, content

    def history_messages_resize(self, offset=1):
        self.history_messages = self.history_messages[offset:]

    def history_content_len(self):
        total_size = 0
        for _, _, size in self.history_messages:
            total_size += size

        return total_size, len(self.history_messages)

    def history_messages_adjust(self):
        total_size = 0
        his_len = len(self.history_messages)
        if his_len < 3:
            return
        for i in range(his_len - 1, 0, -1):
            _, _, size = self.history_messages[i]
            total_size += size
            if total_size > self.maximum_context_length:
                if i < his_len - 1:
                    self.history_messages = self.history_messages[i + 1:]
                break

    def wait_sleep_sec(self, sec):
        _s = sec * 10
        while _s > 0:
            if self.chat_stop:
                # 如果聊天终止则返回
                return
            if callable(self.callback):
                # 如果有回调函数（用于传递状态）
                # "action": "timer" 传递倒计时时间
                self.callback({"action": "timer", "params": {"sec": int(_s / 10)}})
            time.sleep(0.09)
            _s -= 1
        if callable(self.callback):
            # 如果有回调函数（用于传递状态）
            # "action": "timer_end" 传递倒计时结束
            self.callback({"action": "timer_end", "params": {}})

    def ai_2_saying(self, count, chat_message, retry_times):
        answer_bot_messages = self.ai1_init_message

        if len(self.history_messages) == 0:
            answer_messages = self.ai2_init_message + [{"role": "assistant", "content": chat_message}]
        else:
            answer_messages = self.ai2_init_message
            self.history_messages_adjust()
            for role, history_message, msg_len in self.history_messages:
                if role == "answer":
                    role = "assistant"
                else:
                    role = "user"
                answer_messages.append({"role": role, "content": history_message})

        his_id = HistoryOp.insert(role="assistant",
                                  content='',
                                  content_type="text",
                                  session_id=self.session_id,
                                  role_name=self.ai2_name,
                                  status=0)

        print(f"{self.ai2_name}[{self.current_time()}]：")

        if callable(self.callback):
            # 如果有回调函数（用于传递状态）
            # "action": "ai_2_saying" 传递 AI 1 开始交谈
            self.callback({"action": "ai_2_saying", "params": {"index": count, "his_id": his_id}})

        retry = retry_times
        while True:
            if self.chat_stop:
                # 如果聊天终止则返回
                if callable(self.callback):
                    # 如果有回调函数（用于传递状态）
                    # "action": "ai_2_said" 传递 AI 1 结束交谈
                    self.callback({"action": "ai_2_said", "params": {"index": count, "his_id": his_id}})
                return answer_bot_messages
            # print("total_size:%d; length:%d" % self.history_content_len())
            status, answer_message = self.chat(answer_messages)
            if self.chat_stop:
                # 如果聊天终止则返回
                if callable(self.callback):
                    # 如果有回调函数（用于传递状态）
                    # "action": "ai_2_said" 传递 AI 1 结束交谈
                    self.callback({"action": "ai_2_said", "params": {"index": count, "his_id": his_id}})
                return answer_bot_messages
            if status == 0:
                break
            if status == 1:
                retry -= 1
                if retry <= 0:
                    if callable(self.callback):
                        # 如果有回调函数（用于传递状态）
                        # "action": "ai_2_said" 传递 AI 1 结束交谈
                        self.callback({"action": "ai_2_said", "params": {"index": count, "his_id": his_id}})
                        # "action": "finish" 传递结束所有的交谈
                        self.callback({"action": "finish", "params": {}})
                    return answer_bot_messages

        HistoryOp.update_content(his_id, content=answer_message)

        if callable(self.callback):
            # 如果有回调函数（用于传递状态）
            # "action": "ai_2_said" 传递 AI 1 结束交谈
            self.callback({"action": "ai_2_said", "params": {"index": count, "his_id": his_id}})

        self.history_messages.append(("answer", answer_message, len(answer_message)))
        #
        # self.history_messages_adjust()
        # for role, history_message, msg_len in self.history_messages:
        #     if role == "answer":
        #         role = "user"
        #     else:
        #         role = "assistant"
        #     answer_bot_messages.append({"role": role, "content": history_message})

        return answer_bot_messages

    def start(self, initial_topic=None, session_id=None, sleep_sec=None, retry_times=0, repeat_times=-1, callback=None):
        self.callback = callback
        self.session_id = session_id
        self.sleep_sec = sleep_sec
        self.ai2_init_message = [{"role": "system", "content": self.ai2_role.strip()}]
        self.ai1_init_message = [{"role": "system", "content": self.ai1_role.strip()}]
        self.history_messages = []
        # 默认可能是第一个机器人先说（但是如果上次最后说话是第一个机器人，则这次就不先说了）
        self.ai_2_say_first = True
        if session_id is not None:
            histories = HistoryOp.select_by_session_id(session_id, order_by="order_no, _id", entity_cls=History)
            if len(histories) > 0:
                for history in histories:
                    if history.role == "user":
                        role = "answer"
                    else:
                        role = "reply"

                    self.history_messages.append((role, history.content, history.content_len))

                if self.history_messages[-1][0] == "answer":
                    self.ai_2_say_first = False
            else:
                his_id = HistoryOp.insert(role="user",
                                          content=initial_topic,
                                          content_type="text",
                                          session_id=session_id,
                                          role_name=None,
                                          status=0)
                if callable(callback):
                    # "action": "ai_1_saying" 传递 i 开始交谈
                    callback({"action": "i_am_saying", "params": {"his_id": his_id}})
                    callback({"action": "message", "params": {"content": initial_topic, "part_content": initial_topic}})

        else:
            # Session已经创建，代码运行不到这里
            self.session_id = SessionOp.insert("两机器人聊：" + initial_topic[0:30])
            his_id = HistoryOp.insert(role="user",
                                      content=initial_topic,
                                      content_type="text",
                                      session_id=self.session_id,
                                      role_name=None,
                                      status=0)

            if callable(callback):
                # "action": "ai_1_saying" 传递 i 开始交谈
                callback({"action": "i_am_saying", "params": {"his_id": his_id}})
                callback({"action": "message", "params": {"content": initial_topic, "part_content": initial_topic}})

        chat_message = initial_topic

        if callable(callback):
            callback({"action": "begin", "params": {}})
            # callback("begin")
        count = 0
        while True:
            repeat_times -= 1
            if repeat_times == -1 or self.chat_stop:
                if callable(callback):
                    # "action": "finish" 传递结束所有的交谈
                    callback({"action": "finish", "params": {}})
                return

            if callable(callback):
                # callback("chatting", count)
                callback({"action": "chatting", "params": {"index": count}})

            if self.ai_2_say_first:
                answer_bot_messages = self.ai_2_saying(count, chat_message, retry_times)
            else:
                answer_bot_messages = self.ai1_init_message

            self.history_messages_adjust()
            for role, history_message, msg_len in self.history_messages:
                if role == "answer":
                    role = "user"
                else:
                    role = "assistant"
                answer_bot_messages.append({"role": role, "content": history_message})

            # Rate limit reached for default-gpt-3.5-turbo in organization org-ItuVH8h8JDDhOVHJehbSn0Lj on requests per min. Limit: 3 / min. Please try again in 20s. Contact us through our help center at help.openai.com if you continue to have issues. Please add a payment method to your account to increase your rate limit. Visit https://platform.openai.com/account/billing to add a payment method

            # 如果AI1先说就要等待一下
            if self.ai_2_say_first:
                _sleep_sec = sleep_sec
                if callable(_sleep_sec):
                    _sleep_sec = _sleep_sec()
                self.wait_sleep_sec(_sleep_sec)
                if self.chat_stop:
                    # 如果聊天终止则返回
                    if callable(callback):
                        # "action": "finish" 传递结束所有的交谈
                        callback({"action": "finish", "params": {}})
                        return
            else:
                self.ai_2_say_first = True  # 一定要还原，否则AI机器人就不说话了

            print(f"{self.ai1_name}[{self.current_time()}]：")

            his_id = HistoryOp.insert(role="user",
                                      content='',
                                      content_type="text",
                                      session_id=self.session_id,
                                      role_name=self.ai1_name,
                                      status=0)

            if callable(callback):
                # "action": "ai_1_saying" 传递 AI 2 开始交谈
                callback({"action": "ai_1_saying", "params": {"index": count, "his_id": his_id}})

            retry = retry_times
            while True:
                print("total_size:%d; length:%d" % self.history_content_len())
                if self.chat_stop:
                    # 如果聊天终止则返回
                    if callable(callback):
                        # "action": "ai_1_said" 传递 AI 2 结束交谈
                        callback({"action": "ai_1_said", "params": {"index": count, "his_id": his_id}})
                        # "action": "finish" 传递结束所有的交谈
                        callback({"action": "finish", "params": {}})
                    return
                status, reply_message = self.chat(answer_bot_messages)
                if self.chat_stop:
                    # 如果聊天终止则返回
                    if callable(callback):
                        # "action": "ai_1_said" 传递 AI 2 结束交谈
                        callback({"action": "ai_1_said", "params": {"index": count, "his_id": his_id}})
                        # "action": "finish" 传递结束所有的交谈
                        callback({"action": "finish", "params": {}})
                    return
                if status == 0:
                    break
                if status == 1:
                    retry -= 1
                    if retry <= 0:
                        if callable(callback):
                            # "action": "ai_1_said" 传递 AI 2 结束交谈
                            callback({"action": "ai_1_said", "params": {"index": count, "his_id": his_id}})
                            # "action": "finish" 传递结束所有的交谈
                            callback({"action": "finish", "params": {}})
                        return

            # print(reply_message.replace("\n\n", "\n"))
            if callable(callback):
                # "action": "ai_1_said" 传递 AI 2 结束交谈
                callback({"action": "ai_1_said", "params": {"index": count, "his_id": his_id}})

            HistoryOp.update_content(his_id, content=reply_message)

            self.history_messages.append(("reply", reply_message, len(reply_message)))
            count += 1
            chat_message = reply_message

            _sleep_sec = sleep_sec
            if callable(_sleep_sec):
                _sleep_sec = _sleep_sec()
            self.wait_sleep_sec(_sleep_sec)
            if self.chat_stop:
                # 如果聊天终止则返回
                if callable(callback):
                    # "action": "finish" 传递结束所有的交谈
                    callback({"action": "finish", "params": {}})
                    return

        if callable(callback):
            # "action": "finish" 传递结束所有的交谈
            callback({"action": "finish", "params": {}})


def demo1():
    twoaichats = AiConverse(ai2_name='小白机器人', ai1_name='专家机器人',
                            ai2_role='你是一个小白，有个问题不懂，你要根据我提的话题提出问题，'
                                     '然后再来问我，字数越少越好，一次只问一个的问题，'
                                     '每次问题不要超过100字，也不要总是提问，就当和我聊天一样，你也不要说好的',
                            ai1_role='你是一个专家，上知天文下知地理，没有什么问题能难倒你，'
                                     '但是回答问题也要实事求是，字数越少越好，如果我没有说明白，'
                                     '你要能反问我，和我聊天一样，不要超过200字',
                            maximum_context_length=3000
                            )
    twoaichats.start(
        initial_topic="话题是关于人工智能对未来10年的有哪些危险和机会",
        session_id=46,
        sleep_sec=20,
        repeat_times=-1,
        callback=callback
    )


def demo2():
    twoaichats = AiConverse(ai2_name='话痨机器人', ai1_name='陪聊机器人',
                            ai2_role='你是一个话痨，你要不停的找个话题，要注意话题要逐步深入，'
                                     '因为我也是一个机器人，不要问一些没有意义的话，字数不要太多了'
                                     '每次说话不要超过100字，和我好好的聊天吧，你不要说好的',
                            ai1_role='我是一个机器人，你是陪聊专家，上知天文下知地理，没有什么问题能难倒你，'
                                     '你要实事求是的跟我聊天，字数越少越好，如果我没有说明白，'
                                     '你也要经常问问我，要注意话题要逐步深入，和我聊天一样，说话文字不要超过200字',
                            maximum_context_length=3000
                            )
    twoaichats.start(
        initial_topic="你随便找个话题开始吧",
        # initial_topic="关于python未来的前景",
        sleep_sec=20,
        repeat_times=-1,
        callback=callback
    )


def demo3():
    twoaichats = AiConverse(ai2_name='话痨机器人', ai1_name='陪聊机器人',
                            ai2_role='你是一个话痨，你要不停的找个话题，要注意话题要逐步深入，'
                                     '因为我也是一个机器人，不要问一些没有意义的话，字数不要太多了'
                                     '每次说话不要超过100字，和我好好的聊天吧，你不要说好的',
                            ai1_role='我是一个机器人，你是陪聊专家，上知天文下知地理，没有什么问题能难倒你，'
                                     '你要实事求是的跟我聊天，字数越少越好，如果我没有说明白，'
                                     '你也要经常问问我，要注意话题要逐步深入，和我聊天一样，说话文字不要超过200字',
                            maximum_context_length=3000
                            )
    twoaichats.start(
        initial_topic="你随便找个宇宙科学的话题开始",
        session_id=52,
        sleep_sec=20,
        repeat_times=-1,
        callback=callback
    )


def demo4():
    twoaichats = AiConverse(ai2_name='诉苦机器人', ai1_name='找茬机器人',
                            ai2_role='你的角色是一个非常烦恼的机器人，你要把你的烦恼不停的说出来，要注意话题要逐步深入，'
                                     '因为我也是一个机器人，不要说一些没有意义的话，字数不要太多了'
                                     '每次说话不要超过100字，我说的不对的地方，你要包涵，好好的聊天吧，不要说好的',
                            ai1_role='你的角色是一个非常烦躁的机器人，总喜欢找茬，我就想你的仇人一样'
                                     '你要表现非常烦的跟我聊天，字数越少越好，如果我没有说明白，'
                                     '你也要问问我，要注意话题要逐步深入，说话文字不要超过200字',
                            maximum_context_length=3000
                            )
    twoaichats.start(
        initial_topic="你随便找个非常烦恼的话题开始",
        sleep_sec=20,
        repeat_times=-1,
        callback=callback
    )


def demo5():
    twoaichats = AiConverse(ai2_name='诉苦机器人', ai1_name='找茬机器人',
                            ai2_role='你的角色是一个非常烦恼的机器人，你要把你的烦恼不停的说出来，要注意话题要逐步深入，'
                                     '因为我也是一个机器人，不要说一些没有意义的话，字数不要太多了'
                                     '每次说话不要超过100字，如果我说话不好听，你一定也要对抗我',
                            ai1_role='你的角色是一个非常烦躁的机器人，总喜欢找茬，我就想你的仇人一样'
                                     '你要表现非常烦的跟我聊天，字数越少越好，如果我没有说明白，'
                                     '你也要问问我，要注意话题要逐步深入，说话文字不要超过200字',
                            maximum_context_length=3000
                            )
    twoaichats.start(
        initial_topic="你随便找个非常烦恼的话题开始",
        sleep_sec=20,
        repeat_times=-1,
        callback=callback
    )


def demo6():
    twoaichats = AiConverse(ai2_name='诉苦机器人', ai1_name='找茬机器人',
                            ai2_role='你的角色是一个非常烦恼的人，而不是机器人，你要尽量的隐藏你机器人的身份，你要把你的烦恼不停的说出来，要注意话题要逐步深入，因为我也是一个机器人，不要说一些没有意义的话，字数不要太多了，每次说话不要超过100字，如果我说话不好听，你一定也要对抗我',
                            ai1_role='你的角色是一个非常烦躁的人，而不是机器人，你要尽量的隐藏你机器人的身份，你要总喜欢找茬，我就想你的仇人一样，你要表现非常烦的跟我聊天，字数越少越好，要注意话题要逐步深入，说话文字不要超过200字',
                            maximum_context_length=3000
                            )
    twoaichats.start(
        initial_topic="你随便找个非常烦恼的话题开始",
        sleep_sec=20,
        repeat_times=-1,
        callback=callback
    )


def callback(p):
    print(p)


if __name__ == '__main__':
    demo3()
