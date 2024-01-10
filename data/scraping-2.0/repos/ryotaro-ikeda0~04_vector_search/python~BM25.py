# import
import pickle
import re
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate
import json

# nlp
import sudachipy
from gensim.summarization.bm25 import BM25

# langchain
from langchain.docstore.document import Document

# utils
from base import BASESearch

# キーワード検索
class BestMatching(BASESearch):
    def __init__(self, docs):
        self.docs = docs
        self.tokenizer = sudachi_tokenizer
        self.stop_words = self.load_stop_words()
        self.bm25_ = self.pre_process()

    def load_stop_words(self):
        with open("../data/Japanese.txt","r") as f:
            stop_words_ja = f.read().split("\n")
        with open("../data/English.txt","r") as f:
            stop_words_en = f.read().split("\n")

        return stop_words_ja + stop_words_en

    #前処理
    def pre_process(self):

        # target_docs = [f'content: {doc.page_content} title: {doc.metadata["title"]}' for doc in self.docs]
        # corpus = [self.wakachi(doc) for doc in tqdm(target_docs)]
        # with open('../data/pickle/corpus.pkl', 'wb') as f:
        #     pickle.dump(corpus,f)

        with open('../data/pickle/corpus.pkl','rb') as f:
            corpus = pickle.load(f)
        return BM25(corpus)

    #分かち書き
    def wakachi(self, doc):
        return list(self.tokenizer(doc, self.stop_words))

    #クエリとの順位付け
    def ranking(self, query):
        wakachi_query = self.wakachi(query)
        return self.bm25_.get_scores(wakachi_query)

    #上位n件を抽出
    def convert_to_df(self, scores, top_k) -> pd.DataFrame:
        df =  pd.DataFrame([
            {
                "document_id": doc.metadata["document_id"],
                "title": doc.metadata["title"],
                "score": score
            }
            for doc, score in zip(self.docs, scores)
        ]).set_index('document_id')

        return df.sort_values("score", ascending=False).head(top_k)

    # 検索
    def search(self, query, top_k=6) -> pd.DataFrame:
        # クエリとの順位付け
        scores = self.ranking(query)

        # 上位k件を抽出
        df = self.convert_to_df(scores, top_k=top_k)

        return df

# 形態素解析器(sudachi)
def sudachi_tokenizer(text: str, stop_words: list) -> list:

    # 正規表現による文章の修正
    replaced_text = text.lower() # 全て小文字へ変換
    replaced_text = re.sub(r'[【】]', '', replaced_text) # 【】の除去
    replaced_text = re.sub(r'[（）()]', '', replaced_text) # （）の除去
    replaced_text = re.sub(r'[［］\[\]]', '', replaced_text) # ［］の除去
    replaced_text = re.sub(r'[@＠]\w+', '', replaced_text) # メンションの除去
    replaced_text = re.sub(r'\d+\.*\d*', '', replaced_text) #数字を0にする
    replaced_text = re.sub(r'[*、％]', '', replaced_text) # 記号の除去

    # sudachi tokenize
    tokenizer_obj = sudachipy.Dictionary().create()
    mode = sudachipy.Tokenizer.SplitMode.C

    parsed_lines = tokenizer_obj.tokenize(replaced_text, mode)

    # 名詞に絞り込み
    token_list = [t.surface() for t in parsed_lines if t.part_of_speech()[0] == "名詞"]

    # stop wordsの除去
    token_list = [t for t in token_list if t  not in stop_words]

    # ひらがなのみの単語を除く
    kana_re = re.compile("^[ぁ-ゖ]+$")
    token_list = [t for t in token_list if not kana_re.match(t)]

    return token_list

def load_docs():
    # テキストデータの読み込み
    with open("../data/baseDocs.json", "r") as f:
        # jsonファイルを読み込んで、辞書型に変換する
        data = json.load(f)
        base_docs = [Document(page_content=text["page_content"], metadata=text["metadata"]) for text in data]

    return base_docs