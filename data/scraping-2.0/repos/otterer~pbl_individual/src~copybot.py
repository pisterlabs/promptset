import os
import re

from dotenv import load_dotenv
from langchain import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from notion_fetcher import NotionWeeklyReportFetcher
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from tqdm import tqdm
from utility import ACADEMIC_PERIODS, SlackCallbackHandler

load_dotenv()


class CoPyBot:
    def __init__(self):
        self.app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
        self.register_listeners()

    def create_chain(self, llm):
        system_template = """
        You are an assistant who thinks step by step and includes a thought path in your response.
        Your answers are in Japanese.
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = """
        {month}月の週報の内容を以下のルールに従って要約してね。
        要約というのは、文章全体の中から重要なトピックを見つけ出し、そのトピックごとに要点をまとめ、エッセンスとなる情報だけに絞り込むことだよ。
        長い文章は要約とは言えないから注意してね。
            ＜ルール＞
            ・「活動内容と成果の実績」、「課題と解決策」、「できごと・気づき」の３つの大項目を立て、この大項目ごとに要約を記載すること（これ以外の項目は立てないこと）
            ・上記３つの各項目につき、３箇条ずつ箇条書きで要約を記載すること（４箇条以上は記載しないこと）
            ・一文につき各37文字以内とすること
            ・回答全体の総文字数は330文字以内とすること
            ・回答は12行以内とすること
            ・「今週の活動と成果の実績」という名前の項目は立てないこと

            回答作成に当たっては次の記載例を参考にすること。
            文字数や構成は下記記載例から大きく逸脱しないこと。
            ＜記載例＞
                【活動内容と成果の実績】
                ・モブプログラミングを実施し、APIの使い方を学んだ
                ・スクラム運営方針を決定し、スプリントの設計を行った
                ・マンスリーレビューの指摘事項について振り返りを行った
                【課題と解決策】
                ・スケジュール遅延が繰り返し生じている
                ・まだ悲鳴を上げることに抵抗感が残っている
                ・評価指標が未作成である
                【できごと・気づき】
                ・モブプロにより他のメンバーとの知見共有がスムーズになった
                ・輪読会での学びがスプリントの設計に活用できた
                ・スクラムの精神を学び失敗しても大丈夫な雰囲気の醸成ができた

            ＜以下の行から先が週報の内容。下記を要約すること。＞
            {weekly_reports}
        """
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chat_prompt.input_variables = ["month", "weekly_reports"]

        chain = LLMChain(llm=llm, prompt=chat_prompt)

        return chain

    def get_llm(self, say):
        def say_function(message):
            say(message)

        callback_manager = CallbackManager([SlackCallbackHandler(say_function)]) if self.is_streaming else None

        return ChatOpenAI(temperature=0,
                          openai_api_key=os.environ.get("OPENAI_API_KEY"),
                          model_name="gpt-3.5-turbo",
                          streaming=self.is_streaming,
                          callback_manager=callback_manager)

    def weekly_summary(self, period, month, i, week, say):
        notion_api_key = os.environ.get("NOTION_API_KEY")
        client = NotionWeeklyReportFetcher(notion_api_key)

        weekly_reports = client.fetch_records_for_week(period.quarter, week)

        if not weekly_reports:
            return None

        if self.is_streaming:
            say(f"{month}月第{i + 1}週の週報を要約しています...")

        return self.chain.run(month=month, weekly_reports=weekly_reports)

    def monthly_summary(self, summaries, month, say):
        launch_comment = "各週の内容から１か月分の要約を作成中..."
        print(launch_comment)
        if self.is_streaming:
            say(launch_comment)

        concat_summary = " ".join(summaries)
        monthly_report = self.chain.run(month=month, weekly_reports=concat_summary)
        return monthly_report

    def register_listeners(self):
        @self.app.message(re.compile("(週報要約|マンスリーレビュー作って|たのむ|たのんだ)"))
        def message_streamling_mode_selection(say):
            text = "こんにちは。co-py-bot だよ。\nストリーミングモードで実行する？"
            say(
                blocks=[
                    {
                        "type": "section",
                        "block_id": "section677",
                        "text": {"type": "mrkdwn", "text": text},
                        "accessory": {
                            "action_id": "mode_selection",
                            "type": "static_select",
                            "placeholder": {"type": "plain_text", "text": "選択してください"},
                            "options": [{"text": {"type": "plain_text", "text": "はい"}, "value": "1"}, {"text": {"type": "plain_text", "text": "いいえ"}, "value": "0"}],
                        },
                    }
                ],
                text=text,
            )

        @self.app.action("mode_selection")
        def message_month_selection(body, ack, say):
            ack()
            self.is_streaming = int(body["actions"][0]["selected_option"]["value"])

            text = "マンスリーレビューを作成したい対象月を選んでね。"
            say(
                blocks=[
                    {
                        "type": "section",
                        "block_id": "section678",
                        "text": {"type": "mrkdwn", "text": text},
                        "accessory": {
                            "action_id": "month_selection",
                            "type": "static_select",
                            "placeholder": {"type": "plain_text", "text": "対象月を選択"},
                            "options": [{"text": {"type": "plain_text", "text": f"{month}月"}, "value": str(month)} for month in range(1, 13)],
                        },
                    }
                ],
                text=text,
            )

        @self.app.action("month_selection")
        def make_monthly_review(body, ack, say):
            ack()

            month = body["actions"][0]["selected_option"]["value"]
            period = ACADEMIC_PERIODS[month]
            say(f"了解。{month}月の週報をもとにマンスリーレビュー資料をまとめるね。少し待ってね。")

            self.chain = self.create_chain(self.get_llm(say))

            summaries = []
            for i, week in enumerate(tqdm(period.weeks)):
                weekly_summary = self.weekly_summary(period, month, i, week, say)
                if weekly_summary is not None:
                    summaries.append(weekly_summary)
                    print(weekly_summary)

            monthly_report = self.monthly_summary(summaries, month, say)
            say(monthly_report)
            print("Done!")

    def start(self):
        SocketModeHandler(self.app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    bot = CoPyBot()
    bot.start()
