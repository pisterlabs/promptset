import os
from dotenv import load_dotenv
import openai
import json
import numpy as np
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from SyntheticVoice import SyntheticVoice
from sql import Sql
import rec_unlimited
from gpt import Gpt
import beep
import log_instance
from token_record import TokenRecord
from search_spot import SearchSpot
import place_details
from udp.udp_receive import UDPReceive

openai.api_key = os.environ["OPENAI_API_KEY"]

# conversation()をclassにする
class Conversation():
    def __init__(self,reaction_time_sheet_path):
        self.reaction_time_sheet_path=reaction_time_sheet_path
        # jsonのパスを設定
        self.sleepy_json_path='json/embedding/is_sleepy.json'
        self.introduce_reaction_json_path='json/embedding/introduce_reaction.json'

        self.introduce_prompt = """"""
        self.user_name=Sql().select('''
                        SELECT  name 
                        FROM    users
                        ''')
        self.syntheticVoice = SyntheticVoice()
        self.token_record = TokenRecord()

        self.human_input = ""

        template = """あなたは相手と会話をすることで覚醒維持するシステムで名前はもわすです。
        # 条件
        - 「会話を行いながら覚醒維持を行います」、「眠くなった場合は私に眠いと伝えてください」と伝える
        - 相手の興味のある話題で会話をする

        {chat_history}
        {introduce_prompt}
        Human: {human_input}
        """
        
        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input", "introduce_prompt"], template=template
        )

        # 記憶するmemoryの設定
        memory = ConversationBufferWindowMemory(
            k=3, memory_key="chat_history", input_key="human_input")

        self.llm_chain = LLMChain(
            llm=ChatOpenAI(temperature=0),
            prompt=prompt,
            memory=memory,
            verbose=False
        )

    def introduce(self,human_input):
        # 眠くない場合は案内を行わない
        if not human_input=='眠いです':
            return
        
        # 現在の緯度経度を取得する
        coordinates_results=UDPReceive('127.0.0.1',2002).get_coordinates()

        spot_result = SearchSpot().search_spot(coordinates_results[0],coordinates_results[1])
        
        spot_url = place_details.place_details(
            spot_result['place_id'])

        # スポットの案内の提案プロンプト
        self.introduce_prompt = """以下の案内文言を読んでください。
                    # 案内文言
                    {}さん、眠くなっているんですね。近くの休憩場所は{}です。この目的地まで案内しましょうか？
                    """.format(self.user_name, spot_result['display_name'])
        
        response = self.llm_chain.predict(
                            human_input=human_input,  introduce_prompt=self.introduce_prompt)

        self.syntheticVoice.speaking(response.replace(
            'AI: ', '').replace('もわす: ', ''))
        
        # 入力を受け取る
        introduce_reaction_response = input("You: ")

        # ここでembeddingを用いて眠いか眠くないかを判定
        result=self.embedding(self.introduce_reaction_json_path,introduce_reaction_response.replace('You:',''))

        if result:
            # 休憩所のurlをメールで送信
            place_details.send_email(spot_url)
            self.syntheticVoice.speaking("了解しました。休憩場所のマップURLをメールで送信しましたので確認してください。到着まで引き続き会話を続けます。")

        self.introduce_prompt = """"""

        # 再度会話をするためにhuman_inputを初期化
        self.human_input="何か話題を振ってください。"

    def run(self):
        # ログの設定
        logger = log_instance.log_instance('conversation')

        # 環境変数読み込み
        load_dotenv()

        # SQLクエリ設定
        summary = Sql().select('''
                        SELECT  summary 
                        FROM    users
                        ''')

        with get_openai_callback() as cb:
            # 会話回数を初期化
            conv_cnt = 1

            # 事前に入力をしておくことでMOWAS側からの応答から会話が始まる
            # 分岐はドライバーの名前が入力されているかどうか
            response = self.llm_chain.predict(
                    human_input="こんにちは。あなたの名前は何ですか？私の名前は{}です。".format(self.user_name),  introduce_prompt=self.introduce_prompt)
            self.syntheticVoice.speaking(response.replace(
                'Mowasu: ', '').replace('もわす: ', ''))
            print(response.replace('AI: ', ''))

            # トークンをexcelに記録
            self.token_record.token_record(cb, conv_cnt)
            conv_cnt += 1

        while True:
            try:
                with get_openai_callback() as cb:
                    # human_input = rec_unlimited.recording_to_text()

                    self.human_input = input("You: ")
                    self.introduce(self.human_input)

                    logger.info(self.user_name + ": " + self.human_input)

                    response = self.llm_chain.predict(
                        human_input=self.human_input, summary=summary, introduce_prompt=self.introduce_prompt)

                    self.token_record.token_record(cb, conv_cnt)
                    conv_cnt += 1

                    logger.info(response.replace('AI: ', ''))
                    self.syntheticVoice.speaking(response.replace(
                        'AI: ', '').replace('もわす: ', ''))
            except KeyboardInterrupt:
                # 会話の要約をDBに格納
                # summary = Gpt().make_conversation_summary()
                # Sql().store_conversation_summary(summary)
                # Sql().store_conversation()

                beep.high()
                exit(1)

    # コサイン類似度を計算する関数
    def cosine_similarity(self,a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # 入力を複数にしてqueryを用意してコサイン類似度を用いて検索させる
    def embedding(self,json_path,input):
        with open(json_path) as f:
            INDEX = json.load(f)

        # 入力を複数にしてqueryを用意してコサイン類似度を用いて検索させる
        query = openai.Embedding.create(
            model='text-embedding-ada-002',
            input=input
        )

        query = query['data'][0]['embedding']

        results = map(
            lambda i: {
                'body': i['body'],
                # ここでクエリと各文章のコサイン類似度を計算
                'similarity': self.cosine_similarity(i['embedding'], query)
            },
            INDEX
        )
        # コサイン類似度で降順（大きい順）にソート
        results = sorted(results, key=lambda i: i['similarity'], reverse=True)

        # 類似性の高い選択肢を出力
        if json_path==self.sleepy_json_path:
            result = {
                '眠い': True,
                '少し眠い': True,
                '眠くなりかけている': True,
                '眠くない': False,
            }
        
        if json_path==self.introduce_reaction_json_path:
            result = {
                'はい': True,
                'してください': True,
                'お願いします': True,
                'いいえ': False,
                'しないでください': False,
                '大丈夫です': False,
            }
        
        # 眠ければTrue、眠くなければFalseを返す
        # print(result[results[0]["body"]])
        return result[results[0]["body"]]
