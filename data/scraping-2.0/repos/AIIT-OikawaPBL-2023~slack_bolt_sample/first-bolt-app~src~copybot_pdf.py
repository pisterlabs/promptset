import io
import os
import re
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
import openai
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from openai.embeddings_utils import get_embedding
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer
from pdfminer.pdfparser import PDFSyntaxError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError
from tqdm import tqdm
from utility import SlackCallbackHandler, num_tokens

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


# PDFの各ページの情報を管理するためのデータクラス
@dataclass
class Page:
    page_number: int
    text: str
    embedding: List[float]


class CoPyBotPDF:
    def __init__(self):
        self.slack_app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
        self.register_listeners()
        self.model_name = "gpt-3.5-turbo-0613"
        self.n_trials = 0

    def create_chain(self, llm):
        system_template = """
        You are an assistant who thinks step by step and includes a thought path in your response.
        Your answers are in Japanese.
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = """
        {query}
        回答に当たっては、まず質問に関連する情報が下記に含まれているかどうか検討し、
        含まれていない場合には「分かりません」と回答してね。
        含まれている場合には、以下の文章に基づいて回答してね。
        {pdf_content}
        """
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chat_prompt.input_variables = ["pdf_content", "query"]

        chain = LLMChain(llm=llm, prompt=chat_prompt)

        return chain

    def get_llm(self, say):
        def say_function(message):
            say(message)

        callback_manager = CallbackManager([SlackCallbackHandler(say_function)]) if self.is_streaming else None

        return ChatOpenAI(temperature=0,
                          openai_api_key=os.environ.get("OPENAI_API_KEY"),
                          model_name=self.model_name,
                          streaming=self.is_streaming,
                          callback_manager=callback_manager)

    def register_listeners(self):
        @self.slack_app.message(re.compile("(PDF|pdf喰ってね|よろしく)"))
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

        @self.slack_app.action("mode_selection")
        def message_ryokai(body, ack, say):
            ack()
            self.is_streaming = int(body["actions"][0]["selected_option"]["value"])
            if self.is_streaming:
                say("了解。ストリーミングモードで実行するね。\n準備はできたよ。要約して欲しいPDFファイル(.pdf)を投稿してね。")
            else:
                say("了解。ストリーミングモードはオフで実行するね。\n準備はできたよ。要約して欲しいPDFファイル(.pdf)を投稿してね。")

        @self.slack_app.message(re.compile("(質問だよ|聞くね)"))
        def recieve_question_about_pdf(body, say):
            say("なるほど。良い質問だね。回答を考えるからちょっと待ってね。")
            query = body["event"]["text"]
            print("query: ", query)
            self.answer_about_pdf(query, say)

        @self.slack_app.event("message")
        def handle_file_share_events(body, say, client):
            # eventのsubtypeがfile_shareでない場合、メソッドを抜ける
            if "subtype" not in body["event"] or body["event"]["subtype"] != "file_share":
                return
            try:
                self.is_streaming
            except AttributeError:
                self.is_streaming = 0
            say("ファイルを受け取ったよ。なかなか良いドキュメントだね。まずは内容を頭に入れるからちょっと待ってね。")
            event = body["event"]
            try:
                self.process_file_share(event, say, client)
            except KeyError:
                pass

    def init_voronoi_indexer(self):
        """
        クラスタリングによってデータ空間をボロノイ領域に分割することにより
        高速な近傍探索を可能にするためのindexerを初期化するメソッド
        ボロノイ領域の紹介はこちら --> https://ja.wikipedia.org/?curid=91418
        このアルゴリズムでは質問クエリに対して、その周辺領域のみで探索できるので
        総当たりせずに済むので探索にかかる時間を大幅に短縮できる
        """

        # 30ページを超える分量の多いPDFファイルに適用
        n_pages = len(self.embedding_array)
        if n_pages > 30:
            # ボロノイ領域の数
            # クラスタリング空間を定義する量子化器（Quantizer）つくる（L2ノルム）
            self.quantizer = faiss.IndexFlatL2(1536)
            # 下記引数の1536はembeddingの次元数、20はボロノイ領域の数（ボロノイの数は増やすと精度上がって処理時間増えるトレードオフ）
            # embeddingの次元数は`text-embedding-ada-002`以外を使う場合には変更が必要。BERT-baseなら768次元など。
            self.indexer = faiss.IndexIVFFlat(self.quantizer, 1536, 20)
            # ベクトルデータベースからボロノイ領域を生成
            self.indexer.train(self.embedding_array)
            # ボロノイ領域にデータを追加
            self.indexer.add(self.embedding_array)
        else:
            # 分量の少ないテキストデータに対しては総当たり探索でも問題ないので
            # 単体のL2ノルムアルゴリズムを使う。
            self.indexer = faiss.IndexFlatL2(1536)
            # データを追加
            self.indexer.add(self.embedding_array)

    def voronoi_diagram_search(self, query_embedding):
        """
        ボロノイ領域による近傍探索を実行するメソッド

        Args:
            query_embedding (np.ndarray): 質問クエリをembeddingしたやつ

        Returns:
            top_n_pages (pandas.DataFrame): 上位n件のテキスト情報を格納したデータフレーム
        """
        # 近傍探索の実行。
        query_embedding = np.array([query_embedding]).astype("float32")
        _, idx = self.indexer.search(query_embedding, 3)
        return self.pages_df.iloc[idx[0]]

    def answer_about_pdf(self, query, say):
        # query文字列とそのembedding
        query_embedding = get_embedding(query, engine="text-embedding-ada-002")

        # 初回応答時のみボロノイ探索のためのindexerを初期化
        if self.n_trials == 0:
            self.init_voronoi_indexer()

        # ボロノイ探索を実行。上位3件のテキストを取得
        top_n_pages = self.voronoi_diagram_search(query_embedding)

        # 上位n件のテキストを取得してLLMに渡すためのリストに追加
        similarity_texts = []
        for index, row in top_n_pages.iterrows():
            print(f"\nPage {row.page_number}):")
            similarity_texts.append(row.text)

        # https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb
        SEPARATOR = "\n* "
        organized_text = ""
        for s in similarity_texts:
            organized_text += SEPARATOR
            organized_text += s

        if num_tokens(organized_text, self.model_name) > 3000:
            # トークン数が上限にかかっている場合、強制的に文字数を減らす
            # 少し余裕をみて3000文字までに設定
            organized_text = organized_text[:3000]

        self.chain = self.create_chain(self.get_llm(say))
        summary = self.chain.run(pdf_content=organized_text, query=query[5:])
        say(summary)
        self.n_trials += 1

    def process_file_share(self, event, say, client):
        try:
            file_info = client.files_info(file=event["files"][0]["id"]).data["file"]
            if file_info["name"][-4:] == ".pdf":
                file_url = file_info["url_private_download"]
                header = {"Authorization": f"Bearer {os.environ.get('SLACK_BOT_TOKEN')}"}
                response = requests.get(file_url, headers=header)
                if response.status_code != 200:
                    print(f"Failed to download file: status code {response.status_code}")
                    print(f"Response body: {response.text}")
                    return
                try:
                    pdf_file = io.BytesIO(response.content)
                    pdf_text = extract_text(pdf_file)
                    print(f'{"#" * 10} PDFの内容（冒頭100文字） {"#" * 10}\n', pdf_text[:100], f'\n{"#" * 46}')
                except PDFSyntaxError:
                    print("Unable to parse the PDF file.")

                pages_text = self.extract_page_text(pdf_file)

                pages = []
                self.embedding_array = []
                for idx, page_text in enumerate(tqdm(pages_text)):
                    # https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb
                    # ベクトル埋め込みを行うためのOPENAI APIの関数
                    embedding = get_embedding(page_text, engine="text-embedding-ada-002")
                    # ページごとの情報をPageクラスのインスタンスに変換してリストに追加
                    page = Page(page_number=idx + 1, text=page_text, embedding=embedding)
                    pages.append(page)
                    self.embedding_array.append(embedding)

                # pagesリストをDataFrameに変換
                self.pages_df = pd.DataFrame(pages)
                self.embedding_array = np.array(self.embedding_array).astype("float32")
                say("OK. 準備ができたよ。このPDFに関することなら何でも聞いてね。")

            else:
                say("アップロードされたファイルが.pdf 形式じゃないみたい。ちょっとpdfファイル以外読みたくない気分なんだ～。")

        except SlackApiError as e:
            print(f"Error getting file info: {e}")

    def extract_page_text(self, pdf_file):
        pages_text = []
        # extract_pages関数で各ページのレイアウトを表すLTPageオブジェクトのリストを取得
        for page_layout in extract_pages(pdf_file):
            # ページごとのテキスト内容
            single_page_text = ""
            for element in page_layout:
                # LTTextContainer --> テキストを含むレイアウト要素を表す基本クラス
                # isinstance --> elementがLTTextContainerのインスタンスかどうかを判定
                if isinstance(element, LTTextContainer):
                    # get_text() --> elementからテキストを取得
                    # 改行を置換して`single_page_text`に順位追加していくことで
                    # chunkデータに変換している
                    single_page_text += element.get_text().replace("\n", " ")
            # １ページ１行のchunkデータになるようリストに追加
            pages_text.append(single_page_text)
        return pages_text

    def start(self):
        SocketModeHandler(self.slack_app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    bot = CoPyBotPDF()
    bot.start()
