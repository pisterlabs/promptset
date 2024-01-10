"""
アプリケーションの起点となるモジュール。CUIバージョン。
"""

from src.talker import Talker
from src.talker_type import TalkerType
from src.command_handler import CommandHandler
from src.session_factory import SessionFactory
from src.cui import CUI
from src import log_writer
import asyncio
import openai
import os
from dotenv import load_dotenv


class Main:
    def __init__(self) -> None:
        """
        アプリケーションの初期化を行う。
        """

        # APIキーを.envから読み込む。
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        # OpenAIのAPIキーが設定できたか確認し、設定されていない場合は例外を返す
        if not openai.api_key:
            raise ValueError("APIKey is not set.")

        # system_talkerはシステムとしての発言を行うためのオブジェクト。
        self.system_talker = Talker(TalkerType.system, "system", "System")

        # CUIモードのインターフェースを生成する。
        self.view = CUI(self.system_talker)

        log_writer.initialize()

        self.app_initializer = SessionFactory(self.view, self.system_talker)


    async def run(self, start_as_latest_setting: bool = False) -> None:
        """
        アプリケーションのメインループ。
        """
        # アプリケーションのモードを選択する。
        try:
            if start_as_latest_setting:
                session = await self.app_initializer.load_session_from_config()
            else:
                session = await self.app_initializer.ask_app_mode()
        except Exception:
            raise

        # 会話コントローラおよびコマンドハンドラを生成する。
        self.command_handler = CommandHandler(self.system_talker, session)

        # 会話ロジックループを開始。
        try:
            await session.begin()
        finally:
            log_writer.saveJson()
            await self.ask_session_end()


    async def ask_session_end(self) -> None:
        """
        セッション終了処理をユーザーに尋ねる。
        """
        end_types = {
            "N": "新規セッション",
            "R": "同様の設定で新規セッション",
            "Q": "アプリケーションを終了"
        }
        message_text = "=== セッションを終了しました ===\n"
        for key, value in end_types.items():
            message_text += f"    ({key}) {value}\n"
        self.view.print_message_as_system(message_text)
        self.view.enable_user_input()
        user_input = ""
        while user_input not in end_types:
            try:
                user_input = await self.view.request_user_input()
                user_input = user_input.upper()
            except asyncio.CancelledError:
                raise
        message_text = f"{end_types[user_input]}を選択しました。"
        self.view.print_message_as_system(message_text)
        if user_input == "N":
            await self.run()
        elif user_input == "R":
            await self.run(True)
        else:
            return


if __name__ == "__main__":
    main = Main()
    asyncio.run(main.run())
