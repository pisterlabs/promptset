from .abstract_ui import AbstractUI
from .talker import Talker
from .chat_message import ChatMessage
from .user import User
from .gptBot import GPTBot
from .session import Session, SessionConfig, SessionType, SessionConfigLoader
from .auto_task_session import AutoTaskSession
from .agi_session import AgiSession
from .one_on_one_session import OneOnOneSession
from .bot_on_bot_session import BotOnBotSession
from .translater import Translater, GptTranslater, DeepLFreeTranslater, TranslateType
from dotenv import load_dotenv
import openai
import yaml
import os
import asyncio



class SessionFactory:
    """
    セッションを生成するファクトリクラス
    """
    def __init__(self, view: AbstractUI, system_talker: Talker) -> None:
        self.view = view
        self.system_talker = system_talker

        self.DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")
        self.DEEPL_FREE_API_KEY = os.getenv("DEEPL_FREE_API_KEY", "")


    async def ask_app_mode(self) -> Session:
        # session_config.yamlがあるかどうかを確認し、あるならそれを読み込む。
        if await self.ask_load_last_session_config():
            session = await self.load_session_from_config()
            return session
        
        session_type = await self.ask_session_type()
        if session_type == SessionType.one_on_one:
            return await self.make_one_on_one_session()
        elif session_type == SessionType.bot_on_bot:
            return await self.make_bot_on_bot_session()
        elif session_type == SessionType.auto_task:
            return await self.make_auto_task_session()
        elif session_type == SessionType.agi:
            return await self.make_agi_session()
        else:
            raise ValueError("Invalid session type.")


    async def make_one_on_one_session(self, bot_name: str = "", translate_type: TranslateType = TranslateType.undefined) -> Session:
        """
        1対1のセッションを作成する。
        """
        if translate_type == TranslateType.undefined:
            translate_type = await self.ask_translate_mode()
        user = User(self.view)
        if not bot_name:
            bot = await self.ask_bot_select()
        else:
            bot = GPTBot(bot_name, self.system_talker)
        session = OneOnOneSession(self.view, self.system_talker, [user, bot], translate_type)

        if translate_type != TranslateType.none:
            translater = self.pick_translater(translate_type)
            session.set_translater(translater)
        return session

    async def make_bot_on_bot_session(self, participant_names: list[str] = [], translate_type: TranslateType = TranslateType.undefined) -> Session:
        """
        Bot同士のセッションを作成する。
        """
        if translate_type == TranslateType.undefined:
            translate_type = await self.ask_translate_mode()
        if not participant_names:
            bot0 = await self.ask_bot_select(0)
            bot1 = await self.ask_bot_select(1)
        else:
            bot0 = GPTBot(participant_names[0], self.system_talker)
            bot1 = GPTBot(participant_names[1], self.system_talker)
        session = BotOnBotSession(self.view, self.system_talker, [bot0, bot1], translate_type)

        if translate_type != TranslateType.none:
            translater = self.pick_translater(translate_type)
            session.set_translater(translater)
        return session


    async def make_auto_task_session(self) -> Session:
        """
        botが自動でタスクを行うセッションを作成する。
        """
        session = AutoTaskSession(self.view, self.system_talker)
        return session
    
    async def make_agi_session(self, translate_type: TranslateType = TranslateType.undefined) -> Session:
        """
        AGIセッションを作成する。
        """
        if translate_type == TranslateType.undefined:
            translate_type = await self.ask_translate_mode()
        user = User(self.view)
        session = AgiSession(self.view, self.system_talker, user, translate_type)
        if translate_type != TranslateType.none:
            translater = self.pick_translater(translate_type)
            session.set_translater(translater)
        return session


    async def ask_load_last_session_config(self) -> bool:
        """
        session_configの設定を読み込むか尋ねる。
        """

        # session_config.yamlがない場合はFalseを返す。
        if not os.path.exists("session_config.yaml"):
            return False

        text = "前回のセッション設定で始めますか？（y/n）"
        self.view.print_message(ChatMessage(text, self.system_talker.sender_info, False))
        while True:
            answer = await self.view.request_user_input()
            if answer == "y":
                return True
            elif answer == "n":
                return False
            else:
                self.view.print_message(ChatMessage("yかnを入力してください。", self.system_talker.sender_info, False))
            
    
    async def load_session_from_config(self) -> Session:
        """
        session_config.yamlから設定を読み込む。
        """
        session_type = SessionType.none
        with open("session_config.yaml", "r", encoding="utf-8") as infile:
            session_config: SessionConfig = yaml.load(infile, Loader = SessionConfigLoader)
            # セッションタイプを取得する。
            session_type = session_config.session_type
        # セッションタイプに応じて、セッションを作成する。
        if session_type == SessionType.one_on_one:
            return await self.make_one_on_one_session(session_config.participant_names[1], session_config.translate_type)
        elif session_type == SessionType.bot_on_bot:
            return await self.make_bot_on_bot_session(session_config.participant_names, session_config.translate_type)
        elif session_type == SessionType.auto_task:
            return await self.make_auto_task_session()
        elif session_type == SessionType.agi:
            return await self.make_agi_session(session_config.translate_type)
        else:
            raise ValueError("Invalid session type.")
        


    async def ask_session_type(self) -> SessionType:
        session_types = {
            "O": ("One-on-One", SessionType.one_on_one),
            "B": ("Bot-on-Bot", SessionType.bot_on_bot),
            "G": ("自動タスク処理", SessionType.auto_task),
            "A": ("AGI", SessionType.agi),
            "Q": ("終了", SessionType.cancel)
        }

        message_text = "セッションタイプを選択してください：\n"
        for key, values in session_types.items():
            message_text += f"    ({key}) {values[0]}\n"

        message = ChatMessage(message_text, self.system_talker.sender_info, False)
        self.view.print_message(message)

        user_input = ""
        while user_input not in session_types:
            try:
                user_input = await self.view.request_user_input()
                user_input = user_input.upper()
            except asyncio.CancelledError:
                raise
        message_text = f"{session_types[user_input][0]}を選択しました。"
        message = ChatMessage(message_text, self.system_talker.sender_info, False)
        self.view.print_message(message)
        return session_types[user_input][1]

        
    async def ask_bot_select(self, num: int = 0) -> GPTBot:
        text = "会話したいBotの名前を入力してください" if num == 0 else "応答するBotの名前を入力してください"
        message = ChatMessage(text, self.system_talker.sender_info, False)
        self.view.print_message(message)
        directory = "personas/"
        persona_name = ""
        bot_confirm: bool = False
        while(not bot_confirm):
            try:
                persona_name = await self.view.request_user_input()
                file_name = persona_name + ".json"
                filepath = os.path.join(directory, file_name)
                bot_confirm = os.path.exists(filepath)
            except asyncio.CancelledError:
                raise
        bot = GPTBot(persona_name, self.system_talker)
        message = ChatMessage(f"{persona_name}を選択しました。", self.system_talker.sender_info, False)
        self.view.print_message(message)
        return bot
    

    async def ask_translate_mode(self) -> TranslateType:
        translate_types = {
            "N": ("翻訳なし", TranslateType.none),
            "D": ("DeepL翻訳 Free版", TranslateType.deepl_free),
            "G": ("ChatGPT翻訳", TranslateType.chatgpt),
        }

        message_text = "翻訳モードを選択してください：\n"
        for option_key, option_values in translate_types.items():
            message_text += f"    ({option_key}) {option_values[0]}\n"

        message = ChatMessage(message_text, self.system_talker.sender_info, False)
        self.view.print_message(message)

        user_input = ""
        while user_input not in translate_types:
            try:
                user_input = await self.view.request_user_input()
                user_input = user_input.upper()
            except asyncio.CancelledError:
                raise
            
        message_text = f"{translate_types[user_input][0]}を選択しました。"
        message = ChatMessage(message_text, self.system_talker.sender_info, False)
        self.view.print_message(message)
        return translate_types[user_input][1]


    def pick_translater(self, translate_mode: TranslateType) -> Translater:
        if translate_mode == TranslateType.deepl_free and self.DEEPL_FREE_API_KEY:
            return DeepLFreeTranslater(self.DEEPL_FREE_API_KEY)
        elif translate_mode == TranslateType.deepl_free and not self.DEEPL_FREE_API_KEY:
            message = ChatMessage("DeepL APIキーが設定されていません。ChatGPT翻訳で実行します。", self.system_talker.sender_info, False)
            self.view.print_message(message)
            return GptTranslater(self.system_talker)
        elif translate_mode == TranslateType.chatgpt:
            return GptTranslater(self.system_talker)
        else:
            raise ValueError("未定義の翻訳者")