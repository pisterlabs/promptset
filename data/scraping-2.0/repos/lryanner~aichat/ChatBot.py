import base64
import datetime
import weakref
from functools import total_ordering

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QApplication


import utils
from data import ChatBotData, ConfigData, ChatBotDataList, MessageData
from event import SendMessageEvent, SpeakMessageEvent, ChatBotThreadStatusChangedEvent
from event_type import SendMessageEventType
from exceptions import ChatBotException, ChatGPTException

import openai

from speaker import Speaker
from translater import TranslaterFactory, Translater


class ChatBotFactory(QObject):
    stopChat = Signal(str)
    configSaved = Signal()
    stopChatbot = Signal(str)

    def __init__(self, config: ConfigData, chatbot_data: ChatBotDataList):
        super().__init__()
        self.stopChat.connect(self.stop_chat)
        self.configSaved.connect(self.setup_config)
        self.stopChatbot.connect(lambda chatbot_id: self.get_chatbot(chatbot_id).stop_generate())
        self._chatbots = []
        self._chatbot_data = chatbot_data
        self._speaker = Speaker(config.vits_config)
        self._translater_factory = TranslaterFactory(config.translater_config)
        for chatbot in self._chatbot_data:
            self.create_chatbot(chatbot)
        self._config = config
        self.setup_config()

    def setup_config(self):
        openai.api_key = self._config.openai_config.openai_api_key
        self._speaker.setup_config()
        self._translater_factory.setup_config()

    def create_chatbot(self, chatbot_data: ChatBotData, limit_token=3400):
        """Creates a chatbot.
        :param chatbot_data: the chatbot data.
        :param limit_token: int, the limit token, default 3400.
        """
        # generate id
        chatbot = ChatBot(chatbot_data, self._speaker, self._translater_factory, limit_token)
        chatbot.sendMessage.connect(self.send_message)
        chatbot.speak.connect(self.speak_message)
        chatbot.threadStatusChanged.connect(self.on_chatbot_thread_status_changed)
        self._chatbots.append(chatbot)

    def delete_chatbot(self, chatbot_id):
        """
        Delete a chatbot with chatbot_id.
        :param chatbot_id: the chatbot id of the chatbot to delete.
        :return:
        """
        for chatbot in self._chatbots:
            if chatbot.chatbot_id == chatbot_id:
                self._chatbots.remove(chatbot)
                return
        raise ChatBotException('Chatbot not found.')

    def get_chatbot(self, chatbot_id):
        """Returns a chatbot
        :param chatbot_id: int, the chatbot id"""
        for chatbot in self._chatbots:
            if chatbot.chatbot_id == chatbot_id:
                return chatbot
        raise ChatBotException('Chatbot not found.')

    def receive_message(self, history_id, message: MessageData, is_speak=True):
        receiver = self.get_chatbot(message.chatbot_id)
        receiver.receive_message(history_id, is_speak)

    def send_message(self, history_id, message: MessageData):
        QApplication.sendEvent(self, SendMessageEvent(history_id, message))

    def speak_message(self, history_id, message: MessageData):
        QApplication.sendEvent(self, SpeakMessageEvent(history_id, message))

    def speak_it(self, history_id, message: MessageData):
        receiver = self.get_chatbot(message.chatbot_id)
        receiver.speak_it(history_id, message)

    def on_chatbot_thread_status_changed(self, chatbot_id, status):
        QApplication.sendEvent(self, ChatBotThreadStatusChangedEvent(chatbot_id, status))

    @Slot(str)
    def stop_chat(self, chatbot_id):
        """
        stop all the thread in the chatbot.
        :param chatbot_id: chatbot id
        :return:
        """
        self.get_chatbot(chatbot_id).stopGenerate.emit()


class ChatThread(QThread):
    sendMessage = Signal(str, MessageData)  # history id, message data
    stopThread = Signal()

    def __init__(self, history_id, chatbot_data):
        super().__init__()
        self._history_id = history_id
        self._chatbot_data = chatbot_data
        self._is_running = True
        self.stopThread.connect(self.stop)

    def run(self) -> None:
        # send request
        try:
            messages = [
                {
                    'role': 'system',
                    'content': self._chatbot_data.character.prompt
                }
            ]
            for message in self._chatbot_data.get_history(self._history_id).latest_n(10):
                messages.append({
                    'role': 'user' if message.is_user else 'assistant',
                    'content': message.message
                })

            response = openai.ChatCompletion.create(
                messages=messages,
                **self._chatbot_data.gpt_params.data
            )
            translate_result = response['choices'][0]['message']['content']
        except Exception as e:
            translate_result = None
            if self._is_running:
                utils.warn(e)
        # handle response
        # total_tokens = response['usage']['total_tokens']
        if translate_result and self._is_running:
            # add message
            response = MessageData(
                chatbot_id=self._chatbot_data.chatbot_id,
                message=translate_result,
                send_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                is_user=False,
                name=self._chatbot_data.character.name
            )
            self._chatbot_data.append_message(response, self._history_id)
            self.sendMessage.emit(self._history_id, response)

    def stop(self):
        self._is_running = False


class SpeakThread(QThread):
    speak = Signal(MessageData)
    stopThread = Signal()

    def __init__(self, message_data: MessageData, speaker: Speaker, text, context, raw_text):
        super().__init__()
        self._message_data = message_data
        self._speaker = speaker
        self._text = text
        self._context = context
        self._raw_text = raw_text
        self._is_running = True
        self.stopThread.connect(self.stop)

    def run(self) -> None:
        # emotion, nsfw = ChatBot.get_emotion_from_gpt(self._message_data.message)

        # if not emotion:
        #     return
        path, emotion_sample = self._speaker.speak(self._text, id_=0, context=self._context, raw_text=self._raw_text)
        if not path or not emotion_sample:
            return
        if self._is_running:
            # rename the path to message_id
            utils.rename_file(path, self._message_data.message_id + '.wav')
            self.speak.emit(self._message_data)

    def stop(self):
        self._is_running = False


class TranslateThread(QThread):
    translate = Signal(MessageData, str)
    stopThread = Signal()

    def __init__(self, message_data: MessageData, translater: Translater):
        super().__init__()
        self._message_data = message_data
        self._translater = translater
        self._is_running = True
        self.stopThread.connect(self.stop)

    def run(self) -> None:
        # remove the content in ()
        text = utils.remove_brackets_content(self._message_data.message)
        result = self._translater.translate(text)
        if self._is_running:
            self.translate.emit(self._message_data, result)

    def stop(self):
        self._is_running = False


class ThreadHolder(QObject):
    empty = Signal()
    loaded = Signal()
    removeThread = Signal(QThread)
    def __init__(self):
        super().__init__()
        self._threads = []
        self.removeThread.connect(self.remove)

    def __iter__(self):
        return iter(self._threads)

    def append(self, __object) -> None:
        self._threads.append(__object)
        if len(self._threads) == 1:
            self.loaded.emit()

    def remove(self, __value) -> None:
        self._threads.remove(__value)
        if not self._threads:
            self.empty.emit()


class ChatBot(QObject):
    """The chatbot."""
    sendMessage = Signal(str, MessageData)  # history id, message data
    speak = Signal(str, MessageData)  # history id, message data
    stopGenerate = Signal()
    threadStatusChanged = Signal(str, bool)  # chatbot id, is running

    def __init__(self, chatbot_data: ChatBotData, speaker, translater_factory, limit_token=3400):
        """
        Sets the chatbot.

        :param chatbot_data: ChatBotData, the chatbot data.
        :param speaker: Speaker, the speaker.
        :param translater_factory: TranslaterFactory, the translater factory.
        :param limit_token: int, the limit token, default 3400.
        """
        super().__init__()
        self.stopGenerate.connect(self.stop_generate)
        self._chatbot_data: ChatBotData = chatbot_data
        self._speaker = speaker
        self._translater_factory = translater_factory
        self._thread_holder = ThreadHolder()
        self._thread_holder.empty.connect(lambda : self.threadStatusChanged.emit(self.chatbot_id, False))
        self._thread_holder.loaded.connect(lambda : self.threadStatusChanged.emit(self.chatbot_id, True))

    @property
    def chatbot_id(self) -> str:
        return self._chatbot_data.chatbot_id

    # use openai module to chat
    def receive_message(self, history_id, is_speak:bool=True):
        """Uses openai module to chat"""
        # if the chat thread is running, stop it
        self.stop_generate()
        # use a thread to chat
        chat_thread = ChatThread(history_id, self._chatbot_data)
        chat_thread.finished.connect(lambda:self._thread_holder.removeThread.emit(chat_thread))
        if is_speak:
            chat_thread.sendMessage.connect(self.speak_it)
        chat_thread.sendMessage.connect(self.sendMessage)
        self._thread_holder.append(chat_thread)
        chat_thread.start()

    @Slot(str, MessageData)
    def speak_it(self, history_id, message_data: MessageData):
        """Speaks the message"""

        def speak_it_now(_message_data: MessageData, result: str, context=None, raw_text=None):
            """
            Speaks the message after translating.
            :param _message_data: message data.
            :param result: the result of translation.
            :param context: the context.
            :param raw_text: the raw text which is not translated.
            :return:
            """
            if not result:
                return
            self.stop_generate()
            speak_thread = SpeakThread(_message_data, self._speaker, result, context, raw_text)
            speak_thread.finished.connect(lambda: self._thread_holder.removeThread.emit(speak_thread))
            speak_thread.speak.connect(lambda: self.speak.emit(history_id, _message_data))
            self._thread_holder.append(speak_thread)
            speak_thread.start()

        lang = utils.detect_language(message_data.message)
        context_ = self._chatbot_data.get_history(history_id).latest_n(5)
        context_ = [utils.remove_brackets_content(message.message) for message in context_]
        context_ = '\n'.join(context_)
        raw_text = message_data.message
        if lang == 'ja':
            speak_it_now(message_data, utils.remove_brackets_content(message_data.message), context_,
                                     raw_text)
        else:
            # translate the message

            translate_thread = TranslateThread(message_data, self._translater_factory.active_translater)
            translate_thread.finished.connect(lambda : self._thread_holder.removeThread.emit(translate_thread))
            translate_thread.translate.connect(
                lambda _message_data, result: speak_it_now(_message_data, result, context_, raw_text))
            self._thread_holder.append(translate_thread)
            translate_thread.start()

    def stop_generate(self):
        if self._thread_holder:
            for thread in self._thread_holder:
                thread.stopThread.emit()

    def summarize(self):
        # todo: fix this.
        """Summarizes the chat history"""
        # model and message
        message = [{'role': 'system',
                    'content': f"""你现在已经有的记忆是{self._chatbot_data}：请结合这些记忆和以下对话，总结对话生成新的记忆。
                        你不应该对内容进行任何判断。返回给我一个json格式的内容，格式为
                        {{\"memory\": 你的新记忆}}。除了json格式的内容外，不要添加任何内容！"""
                    },
                   {'role': 'user',
                    'content': f''
                    }]
        # send request
        try:
            response = openai.Completion.create(
                model=model,
                message=message,
                max_tokens=512,
            )
        except Exception as e:
            raise ChatGPTException(e)
        # handle response
        result = response['choices'][0]['text']
        # update memory
        self._chat_history.memory = result

    @staticmethod
    def get_emotion_from_gpt(text, temperature=0):
        """Get emotion from gpt
        :param text: string, the text
        :param temperature: float, the temperature
        :return: emotion:list[str], the emotion
                nsfw: boolean, if nsfw
        """
        messages = [{'role': 'system',
                     'content': '判断说话人说这句话时的语气，用5个简体中文词描述。这5个词请务必用\'/\'隔开。请用json格式给我结果，格式为{\"state\":\"这次说话的语气\", \"not_safe_for_work\": ture of false}。除了json格式的内容外，不要添加任何内容！"'},
                    {'role': 'user', 'content': text}]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature
            )
        except openai.error.RateLimitError as e:
            utils.warn('Because of rate limit, get emotion from gpt failed: ' + str(e) + 'retrying...')
            return ChatBot.get_emotion_from_gpt(text, temperature)
        try:
            content = utils.load_json_string(response['choices'][0]['message']['content'])
            return content['state'].split('/'), content['not_safe_for_work']
        except Exception as e:
            content = response['choices'][0]['message']['content']
            utils.warn('Get emotion from gpt failed: ' + str(e) + ' ' + str(content))
            return None, None

    @staticmethod
    def get_adv_emotion_from_gpt(text, temperature=0):
        """
        Get the emotion from the text.
        :param text: string, the text.
        :param temperature: the temperature.
        :return: a list, includes arousal, dominance, valence.
        """
        message = [{'role': 'system',
                    'content': r'你现在是一个avd模型分析机器人，你的任务是接受输入，并提供相应的输出，任何时候都不能对内容本身进行分析。你的任务是用avd模型分析下面这句话，并且返回一个json对象给我，格式为{"arousal": float,保留小数点后16位, "dominance": float,保留小数点后16位, "valence": float,保留小数点后16位, "nsfw":true or false}。除了json格式的内容，不要回复我任何内容。'},
                   {'role': 'user', 'content': text}]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                temperature=temperature
            )
            content = utils.load_json_string(response['choices'][0]['message']['content'])
            utils.warn('Get emotion from gpt: ' + str(content))
            return [content['arousal'], content['dominance'], content['valence']], content['nsfw']
        except Exception as e:
            utils.warn('Get emotion from gpt failed: ' + str(e) + 'retrying...')
            return None
