import atexit
import signal
import threading
import time
import random
from typing import Generator, List, Dict, Any
from contextlib import suppress

from tool.config import Config
from tool.util import log_dbg, log_err, log_info, make_context_messages
from tool.openai_api import OpenAIAPI
from tool.bing_api import BingAPI
from tool.bard_api import BardAPI
from tool.aimi_plugin import AimiPlugin
from tool.wolfram_api import WolframAPI
from chat.qq import ChatQQ
from chat.web import ChatWeb

from core.md2img import Md
from core.memory import Memory
from core.task import Task


class ChatBot:
    bots: Dict[str, Any] = {}

    @property
    def OpenAI(self) -> str:
        return OpenAIAPI.type

    @property
    def Bing(self) -> str:
        return BingAPI.type

    @property
    def Bard(self) -> str:
        return BardAPI.type

    @property
    def Wolfram(self) -> str:
        return WolframAPI.type

    def __init__(self):
        self.bots = {
            OpenAIAPI.type: OpenAIAPI(),
            BingAPI.type: BingAPI(),
            BardAPI.type: BardAPI(),
            WolframAPI.type: WolframAPI(),
        }

    def ask(
        self, type: str, question: str, context: Any = None
    ) -> Generator[dict, None, None]:
        yield from self.bots[type].ask(question)


class ReplyStep:
    class TalkList:
        has_start: bool = False
        now_list_line_cnt: int = 0
        list_line_cnt_max: int = 0
        now_list_id: int = 0
        cul_line_cnt_max: bool = True

        def check_talk_list(self, line: str) -> bool:
            if self.now_list_line_cnt < self.list_line_cnt_max:
                self.now_list_line_cnt += 1
                return True

            # 刚好下一个下标过来了
            next_list_id_str = "{}. ".format(self.now_list_id + 1)
            next_list_id_ch_str = "{}。 ".format(self.now_list_id + 1)
            next_list_id_bing_str = "[{}]: ".format(self.now_list_id + 1)
            if (
                (next_list_id_str in line)
                or (next_list_id_ch_str in line)
                or (next_list_id_bing_str in line)
            ):
                log_dbg("check talk list[{}]".format(self.now_list_id))
                self.now_list_line_cnt = 0
                self.now_list_id += 1
                return True

            return False

        def reset(self):
            self.has_start = False
            self.now_list_line_cnt = 0
            self.list_line_cnt_max = 0
            self.now_list_id = 0
            self.cul_line_cnt_max = True

        def is_talk_list(self, line: str):
            # 有找到开始的序号
            if (not self.has_start) and (
                ("1. " in line) or ("1。 " in line) or ("[1]: " in line)
            ):
                self.has_start = True
                self.now_list_line_cnt = 1
                self.list_line_cnt_max = 1
                self.now_list_id = 1
                return True

            # 标记过才处理
            if not self.has_start:
                return False

            if "\n" == line:
                return True

            # 已经找到当前每行的长度
            if not self.cul_line_cnt_max:
                ret = self.check_talk_list(line)
                if not ret:
                    self.reset()
                return ret

            if (self.now_list_id) and (
                ("2. " in line) or ("2。 " in line) or ("[2]: " in line)
            ):
                self.now_list_id = 2
                self.now_list_line_cnt = 0
                self.cul_line_cnt_max = False
                ret = self.check_talk_list(line)
                if not ret:
                    self.reset()
                return ret

            # 统计每块最大行
            self.list_line_cnt_max += 1
            return True

    class MathList:
        has_start: bool = False

        def __is_math_format(md: Md, line: str) -> bool:
            if "=" in line:
                return True
            if md.has_latex(line):
                log_dbg("match: is latex")
                return True
            if md.has_html(line):
                log_dbg("match: is html")
                return True
            return False

        def is_math_list(self, md: Md, line: str) -> bool:
            if self.__is_math_format(md, line):
                self.has_start = True
                return True

            if not self.has_start:
                return False

            if "\n" == line:
                return True

            self.has_start = False
            return False


class Aimi:
    type: str = "Aimi"
    timeout: int = 360
    master_name: str = ""
    aimi_name: str = "Aimi"
    preset_facts: Dict[str, str] = {}
    max_link_think: int = 1024
    running: bool = True
    api: List[str] = []
    config: Config
    md: Md
    memory: Memory
    task: Task
    aimi_plugin: AimiPlugin
    openai_api: OpenAIAPI
    bing_api: BardAPI
    bard_api: BardAPI
    wolfram_api: WolframAPI
    chat_web: ChatWeb
    chat_qq: ChatQQ
    chatbot: ChatBot

    def __init__(self):
        self.__load_setting()
        self.config = Config()
        self.md = Md()
        self.memory = Memory()
        self.aimi_plugin = AimiPlugin()

        self.chatbot = ChatBot()
        self.openai_api: OpenAIAPI = self.chatbot.bots[self.chatbot.OpenAI]
        self.max_link_think = self.openai_api.max_requestion
        self.bing_api = self.chatbot.bots[self.chatbot.Bing]
        self.bard_api = self.chatbot.bots[self.chatbot.Bard]
        self.wolfram_api = self.chatbot.bots[self.chatbot.Wolfram]

        self.task = Task(self.chatbot)

        self.chat_web = ChatWeb()
        self.chat_qq = ChatQQ()

        # 注册意外退出保护记忆
        atexit.register(self.__when_exit)
        signal.signal(signal.SIGTERM, self.__signal_exit)
        signal.signal(signal.SIGINT, self.__signal_exit)

        try:
            self.aimi_plugin.when_init()
        except Exception as e:
            log_err(f"fail to init aimi plugin: {e}")

        self.chat_web.register_ask_hook(self.web_ask)
        self.chat_web.register_get_models_hook(self.get_bot_models)

    def make_link_think(
        self,
        question: str,
        nickname: str = None,
        api_type: str = None,
        preset: str = "",
        talk_history: Any = None,
    ) -> str:
        nickname = nickname if nickname and len(nickname) else self.master_name
        api_type = api_type if api_type and len(api_type) else self.api[0]
        link_think = question

        if api_type == self.task.type:
            link_think = self.task.make_link_think(question, self.aimi_name, preset)
        elif api_type == self.openai_api.type and self.openai_api.use_web_ask:
            # cul question
            history = self.memory.make_history(talk_history)
            link_think = f"""
设定: {{
“{preset}”
}}.

请只关注最新消息,历史如下: {{
{history}
}}.

请根据设定和最新对话历史和你的历史回答, 不用“{self.aimi_name}:”开头, 回答如下问题: {{
{nickname}说: “{question}”
}}.
"""
        elif api_type == self.bard_api.type:
            link_think = f"""
preset: {{
\"{preset}\"
}}.

Please answer the following question based on the preset, 
the latest conversation history, and your previous answers.
and without starting with '{self.aimi_name}:'
You should extract my question directly from the structure here and answer it directly: {{
{nickname} said: '{question}'
}}
"""
        elif "" == self.bing_api.type:
            link_think = f"""
故事主题如下: 故事中有两个人到了异世界里面开咖啡厅.
故事情节如下: 其中主人公自称Master({self.master_name}), 他有一个助理, 名叫{self.aimi_name}, 他们两个在聊天.
故事背景如下: {{
{preset}
{self.aimi_name}能问她的朋友 Bing 来尽力解答Master的问题.
}}

请不显示故事主题/情节/背景的分析过程, 以“{self.aimi_name}”的身份, 让聊天足够自然, 接下以下聊天: {{
{nickname}说: '{question}'
}}
"""
        elif api_type == self.wolfram_api.type:
            math_html = "help me caculate math."
            for answer in self.__post_wolfram(link_think):
                if answer["code"] != 0:
                    continue
                math_html = answer["message"]
            link_think = f"""
{link_think}
需求:
1. 使用 latex 显示内容比如:
' $$ 
x_1 
$$ ' .

2. 我需要你使用翻译功能, 帮我把 the wolfram response 的内容翻译成我的语言.
3. 请忽略历史记录, 你现在是严谨的逻辑分析翻译结构.
4. 你的品味很高,你会使内容更加美观,你会添加足够多的二次换行,并在不同层次追加换行,并按照自己经验让内容变得更加好看,答案要重点突出,并用latex显示.
5. 不要试图解决这个问题，你只能显示 the wolfram response 里面的内容.

the wolfram response: {{
{math_html}
}}
"""
        elif api_type == "chimeragpt":
            # cul question
            history = self.memory.make_history(talk_history)

            link_think = f"""
preset: {{
“{preset}”
}}.

Please focus only on the latest news. History follows: {{
{history}
}}

Based on the preset and our conversation history, and my previous responses, 
answer the following question without adding additional descriptions, 
and without starting with '{self.aimi_name}:'
answer this following question: {{
{nickname} say: “{question}”
}}.
"""
        return link_think

    def run(self):
        self.notify_online()

        aimi_read = threading.Thread(target=self.read)
        chat_qq_server = threading.Thread(target=self.chat_qq.server)
        chat_web_server = threading.Thread(target=self.chat_web.server)
        aimi_dream = threading.Thread(target=self.memory.dream)

        # 同时退出
        aimi_read.setDaemon(True)
        aimi_dream.setDaemon(True)
        chat_qq_server.setDaemon(True)
        chat_web_server.setDaemon(True)

        aimi_read.start()
        aimi_dream.start()
        chat_qq_server.start()
        chat_web_server.start()

        cnt = 0
        while self.running:
            cnt = cnt + 1
            if cnt < 60:
                time.sleep(1)
                continue
            else:
                cnt = 0

            try:
                if not self.memory.save_memory():
                    log_err("save memory failed")
                if not self.task.save_task():
                    log_err("save task failed")

            except Exception as e:
                log_err("fail to save: " + str(e))

        log_dbg("aimi exit")

    def __question_model(self, api_type, question: str) -> str:
        if api_type == self.task.type:
            return self.task.get_model(question)
        return ""

    def __question_api_type(self, question: str) -> str:
        if self.bing_api.is_call(question):
            return self.bing_api.type
        if self.bard_api.is_call(question):
            return self.bard_api.type
        if self.openai_api.is_call(question):
            return self.openai_api.type
        if self.aimi_plugin.bot_is_call(question):
            return self.aimi_plugin.bot_get_call_type(question)
        if self.wolfram_api.is_call(question):
            return self.wolfram_api.type
        if self.task.is_call(question):
            return self.task.type

        return self.api[0]

    @property
    def __busy_reply(self) -> str:
        busy = [
            "让我想想...",
            "......",
            "那个...",
            "这个...",
            "?",
            "喵喵喵？",
            "*和未知敌人战斗中*",
            "*大脑宕机*",
            "*大脑停止响应*",
            "*尝试构造语言中*",
            "*被神秘射线击中,尝试恢复中*",
            "*猫猫叹气*",
        ]
        return random.choice(busy)

    def read(self):
        while self.running:
            if not self.chat_qq.has_message():
                time.sleep(1)
                continue

            for msg in self.chat_qq:
                log_info("recv msg, try analyse")
                nickname = self.chat_qq.get_name(msg)
                question = self.chat_qq.get_question(msg)
                log_info("{}: {}".format(nickname, question))

                api_type = self.__question_api_type(question)

                reply = ""
                reply_line = ""
                reply_div = ""
                answer = {}

                talk_list = ReplyStep.TalkList()
                math_list = ReplyStep.MathList()
                code = 0
                for answer in self.ask(question, nickname):
                    code = answer["code"]

                    message = answer["message"][len(reply) :]
                    reply_line += message

                    reply = answer["message"]

                    reply_div_len = len(reply_div)
                    log_dbg(
                        f"code: {str(code)} div: {str(reply_div_len)} line: {str(reply_line)}"
                    )

                    if code == 0 and (
                        len(reply_div) or ((not len(reply_div)) and len(reply_line))
                    ):
                        reply_div += reply_line
                        reply_line = ""

                        reply_div = self.reply_adjust(reply_div, api_type)
                        log_dbg(f"send div: {str(reply_div)}")
                        self.chat_qq.reply_question(msg, reply_div)

                        break
                    if (code == -1) and (len(reply_div) or len(reply_line)):
                        if not len(reply_div):
                            reply_div = self.__busy_reply
                        reply_div = self.reply_adjust(reply_div, api_type)
                        log_dbg(f"fail: {str(reply_line)}, send div: {str(reply_div)}")
                        self.chat_qq.reply_question(msg, reply_div)
                        reply_line = ""
                        reply_div = ""
                        continue

                    if code != 1:
                        continue

                    if "\n" in reply_line:
                        if talk_list.is_talk_list(reply_line):
                            reply_div += reply_line
                            reply_line = ""
                            continue
                        elif math_list.is_math_list(self.md, reply_line):
                            reply_div += reply_line
                            reply_line = ""
                            continue
                        elif not len(reply_div):
                            # first line.
                            reply_div += reply_line
                            reply_line = ""

                        reply_div = self.reply_adjust(reply_div, api_type)

                        log_dbg("send div: " + str(reply_div))

                        self.chat_qq.reply_question(msg, reply_div)

                        # 把满足规则的先发送，然后再保存新的行。
                        reply_div = reply_line
                        reply_line = ""

                log_dbg(f"answer: {str(type(answer))} {str(answer)}")
                reply = self.reply_adjust(reply, api_type)
                log_dbg(f"adjust: {str(reply)}")

                log_info(f"{nickname}: {question}")
                log_info(f"{self.aimi_name}: {str(reply)}")

                if code == 0:
                    pass  # self.chat_qq.reply_question(msg, reply)

                # server failed
                if code == -1:
                    meme_err = self.config.meme.error
                    img_meme_err = self.chat_qq.get_image_message(meme_err)
                    self.chat_qq.reply_question(msg, "server unknow error :(")
                    self.chat_qq.reply_question(msg, img_meme_err)

                # trans text to img
                if self.md.need_set_img(reply):
                    log_info("msg need set img")
                    img_file = self.md.message_to_img(reply)
                    cq_img = self.chat_qq.get_image_message(img_file)

                    self.chat_qq.reply_question(msg, cq_img)

    def reply_adjust(self, reply: str, res_api: str) -> str:
        if res_api == self.bing_api.type:
            reply = reply.replace("必应", f" {self.aimi_name}通过必应得知: ")
            reply = reply.replace("你好", " Master你好 ")
            reply = reply.replace("您好", " Master您好 ")

        return reply

    def web_ask(
        self,
        question: str,
        nickname: str = None,
        model: str = "auto",
        owned_by: str = "Aimi",
        context_messages: Any = None,
    ) -> Generator[dict, None, None]:
        if (owned_by == self.aimi_name) and (model == "auto"):
            return self.ask(question, nickname)
        elif owned_by == self.aimi_name and (
            model == "task" or model == "task-16k" or model == "task-4k"
        ):
            preset = context_messages[0]["content"]
            task_link_think = self.task.make_link_think(
                question, self.aimi_name, preset
            )
            model = self.task.get_model(model)

            return self.task.ask(task_link_think, model)
        else:
            preset = context_messages[0]["content"]
            talk_history = context_messages[1:]
            link_think = self.make_link_think(
                question=question,
                nickname=self.master_name,
                api_type=owned_by,
                preset=preset,
                talk_history=talk_history,
            )

            return self.__post_question(
                link_think=link_think,
                api_type=owned_by,
                model=model,
                context_messages=context_messages,
            )

    def ask(self, question: str, nickname: str = None) -> Generator[dict, None, None]:
        api_type = self.__question_api_type(question)
        model = self.__question_model(api_type, question)
        nickname = nickname if nickname and len(nickname) else self.master_name

        preset = ""
        with suppress(KeyError):
            preset = self.preset_facts[api_type]
        talk_history = self.memory.search(question, self.max_link_think)

        link_think = self.make_link_think(
            question, nickname, api_type, preset, talk_history
        )
        context_messages = make_context_messages(question, preset, talk_history)

        answer = self.__post_question(
            link_think=link_think,
            api_type=api_type,
            model=model,
            context_messages=context_messages,
        )

        for message in answer:
            if not message:
                continue
            # log_dbg(f'message: {str(type(message))} {str(message)} answer: {str(type(answer))} {str(answer))}'

            # save self.memory
            if message["code"] == 0:
                self.memory.append(q=question, a=message["message"])

            yield message

    def __post_question(
        self, link_think: str, api_type: str, model: str, context_messages: Any
    ) -> Generator[dict, None, None]:
        log_dbg("use api: " + str(api_type))

        if api_type == self.openai_api.type:
            yield from self.__post_openai(
                link_think, model, context_messages, self.memory.openai_conversation_id
            )
        elif api_type == self.task.type:
            yield from self.task.ask(link_think, model)
        elif api_type == self.bing_api.type:
            yield from self.__post_bing(link_think, model)
        elif api_type == self.bard_api.type:
            yield from self.__post_bard(link_think)
        elif self.aimi_plugin.bot_has_type(api_type):
            yield from self.aimi_plugin.bot_ask(
                api_type, link_think, model, context_messages
            )
        elif api_type == self.wolfram_api.type:
            # at mk link think, already set wolfram response.
            if True and self.api[0] != self.wolfram_api.type:
                yield from self.__post_question(
                    link_think,
                    self.api[0],
                    model,
                    context_messages=[{"role": "user", "content": link_think}],
                )
            else:
                yield from self.__post_bing(link_think)
        else:
            log_err("not suppurt api_type: " + str(api_type))

    def __post_wolfram(self, question: str) -> Generator[dict, None, None]:
        yield from self.wolfram_api.ask(question)

    def __post_bard(self, question: str) -> Generator[dict, None, None]:
        yield from self.bard_api.ask(question)

    def __post_bing(
        self,
        question: str,
        model: str = None,
    ) -> Generator[dict, None, None]:
        yield from self.bing_api.ask(question, model)

    def __post_openai(
        self,
        question: str,
        model: str,
        context_messages: List[Dict] = [],
        openai_conversation_id: str = None,
    ) -> Generator[dict, None, None]:
        answer = self.openai_api.ask(
            question, model, context_messages, openai_conversation_id
        )
        # get yield last val
        for message in answer:
            # log_dbg('now msg: ' + str(message))

            try:
                if (
                    (message)
                    and (message["code"] == 0)
                    and message["conversation_id"]
                    and (
                        message["conversation_id"] != self.memory.openai_conversation_id
                    )
                ):
                    self.memory.openai_conversation_id = message["conversation_id"]
                    log_info(
                        "set new con_id: " + str(self.memory.openai_conversation_id)
                    )
            except Exception as e:
                log_dbg(f"no conv_id")

            yield message

    def get_bot_models(self) -> Dict[str, List[str]]:
        bot_models = {}

        bot_models[self.aimi_name] = ["auto", "task", "task-4k", "task-16k"]

        models = self.openai_api.get_models()
        if len(models):
            bot_models[self.openai_api.type] = models

        models = self.bing_api.get_models()
        if len(models):
            bot_models[self.bing_api.type] = models

        models = self.wolfram_api.get_models()
        if len(models):
            bot_models[self.wolfram_api.type] = models

        models = self.bard_api.get_models()
        if len(models):
            bot_models[self.bard_api.type] = models

        plugin_models = self.aimi_plugin.bot_get_models()
        for bot_type, models in plugin_models.items():
            if len(models):
                bot_models[bot_type] = models

        # log_dbg(f"models: {str(bot_models)}")

        return bot_models

    def __load_setting(self):
        try:
            setting = Config.load_setting("aimi")
        except Exception as e:
            log_err(f"fail to load {self.type}: {e}")
            setting = {}
            return

        try:
            self.aimi_name = setting["name"]
        except Exception as e:
            log_err("fail to load aimi: {e}")
            self.aimi_name = "Aimi"
        try:
            self.master_name = setting["master_name"]
        except Exception as e:
            log_err("fail to load aimi: {e}")
            self.master_name = ""

        try:
            self.api = setting["api"]
        except Exception as e:
            log_err("fail to load aimi api: " + str(e))
            self.api = [self.openai_api.type]

        try:
            self.preset_facts = {}
            for api in self.api:
                try:
                    preset_facts: List[str] = setting["preset_facts"][api]
                except Exception as e:
                    log_info(f"no {api} type preset, skip.")
                    continue

                self.preset_facts[api] = ""
                count = 0
                for fact in preset_facts:
                    fact = fact.replace("<name>", self.aimi_name)
                    fact = fact.replace("<master>", self.master_name)
                    count += 1
                    if count != len(preset_facts):
                        fact += "\n"
                    self.preset_facts[api] += fact

            self.preset_facts["default"] = self.preset_facts[self.api[0]]
        except Exception as e:
            log_err("fail to load aimi preset: " + str(e))
            self.preset_facts = {}

    def notify_online(self):
        if not self.chat_qq.is_online():
            log_err(f"{self.chat_qq.type} offline")
            return
        self.chat_qq.reply_online()

    def notify_offline(self):
        self.chat_qq.reply_offline()

    def __signal_exit(self, sig, e):
        log_info("recv exit sig.")
        self.running = False
        self.chat_qq.stop()

    def __when_exit(self):
        self.running = False

        log_info("now exit aimi.")
        self.notify_offline()

        if self.memory.save_memory():
            log_info("exit: save self.memory done.")
        else:
            log_err("exit: fail to save self.memory.")

        if self.task.save_task():
            log_info("exit: save task done.")
        else:
            log_err("exit: fail to task self.memory.")

        try:
            self.aimi_plugin.when_exit()
        except Exception as e:
            log_err(f"fail to exit aimi plugin: {e}")
