import os
import sys

import numpy as np
import openai
import pandas as pd
from copybot_pdf import CoPyBotPDF
from dotenv import load_dotenv
from openai import Embedding
from openai.embeddings_utils import get_embedding

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


class Test(CoPyBotPDF):
    def test(self):
        """
        CoPyBotPDFクラスを継承して動作確認用のメソッドを定義
        """
        # 検索対象となるテキスト群を定義（ChatGPTで適当に生成した文章群）
        self.target_texts = [
            "今日は雨森雅司に賽振られたねえ",
            "小泉今日子は雨後の筍喰えるかな",
            "マンゴー食べれてよかったねえ",
            "良い天気だね～～",
            "キウイはビタミンCが豊富だね",
            "カナダは北米に位置しているよ",
            "アイスランドは温泉が有名だね",
            "サッカーは楽しいスポーツだ",
            "日本は四季がはっきりしているね",
            "バナナは栄養価が高いよ",
            "オーストラリアはコアラがいるよ",
            "バスケットボールは高身長が有利だ",
            "インドはカレーがおいしいね",
            "フランスはワインが有名だよ",
            "ブラジルはサンバの国だね",
            "夏は海水浴が楽しいね",
            "ブルーベリーは目にいいよ",
            "エジプトはピラミッドがあるよ",
            "冬はスキーが楽しいね",
            "パリは芸術の盛んだった都市だよ",
            "春は桜がきれいだね",
            "中国は人口が多い国だよ",
            "フィジーはビーチが素晴らしいね",
            "イタリアはピザが有名だよ",
            "オレンジは甘酸っぱいね",
            "アフリカはサバンナが広がっているよ",
            "夏はバーベキューが楽しいね",
            "アイルランドはビールがおいしいよ",
            "ドイツはビールが有名だね",
            "秋は紅葉がきれいだよ",
            "ロンドンは雨が多いね",
            "日本は寿司が有名だよ",
            "春は新しい出会いがあるね",
            "ブドウはワインの原料だよ",
            "スペインは闘牛が有名だね",
            "冬は雪だるま作りが楽しいよ",
            "トマトは野菜の一種だよ",
            "ハワイはサーフィンが楽しいね",
            "ロシアは広大な国土を持っているよ",
            "森林浴は心地よいね",
            "デンマークは自転車が多いよ",
            "カボチャはハロウィンの象徴だね",
            "ベネズエラは美しいビーチがあるよ",
            "冬は雪合戦が楽しいね",
            "スイスはチーズが美味しいよ",
            "メキシコはタコスが有名だね",
            "南アフリカはサファリが人気だよ",
            "バドミントンは速いラリーがあるね",
            "ポーランドは美しい城があるよ",
            "ソウルはショッピングが楽しいね",
            "ペットは癒しをくれるよ",
            "スウェーデンはIKEAの本国だね",
            "プールは夏に涼しいね",
            "ジャガイモは料理の基本だよ",
            "オランダはチューリップが有名だね",
            "ベルギーはワッフルが美味しいよ",
            "釣りは静かな趣味だね",
            "マラソンは耐久力が試されるよ",
            "シンガポールは夜景がきれいだね",
            "アボカドは健康に良いよ",
            "ノルウェーはフィヨルドが有名だね",
            "チェスは頭の体操だよ",
            "イスラエルは歴史が深いね",
            "スペインはフラメンコが魅力的だよ",
            "ヨガはリラックスに良いね",
            "アメリカはバーガーが人気だよ",
            "イタリアはジェラートが美味しいね",
            "バレエは優雅なダンスだよ",
            "サッカーは国際的なスポーツだね",
            "ドバイは超高層ビルが立つよ",
            "カプサイシンは辛さの元だよ",
            "ニュージーランドは羊が多いね",
            "オリーブオイルは料理に欠かせないよ",
            "オーストラリアはコアラがいるよ",
            "バスケットは高速なゲームだね",
            "エクアドルは赤道が通るよ"] * 3

        # 検索対象となるテキスト群をベクトル化
        response = Embedding.create(input=self.target_texts,
                                    model="text-embedding-ada-002")
        # 色んなプロパティを含んでいるresponseの中から、ベクトル表現のみを取り出したリストを作る
        self.embedding = [record["embedding"] for record in response["data"]]
        # ベクトルのnumpy配列を作成（動作確認を行うメソッドで必要）
        self.embedding_array = np.array(self.embedding).astype("float32")
        # 検索対象となるテキスト群のDataFrameを作成（動作確認を行うメソッドで必要）
        self.pages_df = pd.DataFrame(self.target_texts, columns=["text"])


if __name__ == "__main__":
    query = "今日は雨振らんくてよかったねえ"
    query_embedding = get_embedding(query, engine="text-embedding-ada-002")

    # CoPyBotPDFを継承したTestクラスを作成
    test = Test()

    # 動作確認用のインスタンス変数を作成
    test.test()

    """以下で今回のプルリクで新たに実装したメソッドを実行"""

    # ボロノイ探索のためのindexerを初期化
    test.init_voronoi_indexer()
    # ボロノイ探索を実行。上位3件のテキストを取得
    top_n_pages = test.voronoi_diagram_search(query_embedding)
    print(top_n_pages.iloc[0, 0])
