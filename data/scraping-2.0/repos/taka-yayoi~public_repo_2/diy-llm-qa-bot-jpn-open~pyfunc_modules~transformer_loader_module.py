import re
import time
import mlflow
import torch
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores.faiss import FAISS
from langchain.schema import BaseRetriever
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain import LLMChain


class QABot():

  template = r"<s>\nあなたはDatabricksによって開発された有能なアシスタントであり、指定された文脈に基づいて質問に回答することを得意としており、文脈は文書です。文脈が回答を決定するのに十分な情報を提供しない場合には、わかりませんと言ってください。文脈が質問に適していない場合には、わかりませんと言ってください。文脈から良い回答が見つからない場合には、わかりませんと言ってください。問い合わせが完全な質問になっていない場合には、わからないと言ってください。あなたは同じ言葉を繰り返しません。以下は、文脈を示す文書と、文脈のある質問の組み合わせです。文書を要約することで質問を適切に満たす回答をしなさい。\n[SEP]\n文書:\n{context}\n[SEP]\n質問:\n{question}\n[SEP]\n回答:\n"

  def __init__(self, llm, retriever):
    self.llm = llm
    self.retriever = retriever
    self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])
    self.qa_chain = LLMChain(llm = self.llm, prompt=self.prompt)
    self.abbreviations = { # 置換したい既知の略語
      "DBR": "Databricks Runtime",
      "ML": "Machine Learning",
      "UC": "Unity Catalog",
      "DLT": "Delta Live Table",
      "DBFS": "Databricks File Store",
      "HMS": "Hive Metastore",
      "UDF": "User Defined Function"
      } 


  def _is_good_answer(self, answer):

    ''' 回答が妥当かをチェック '''

    result = True # デフォルトのレスポンス

    badanswer_phrases = [ # モデルが回答を生成しなかったことを示すフレーズ
      "わかりません", "コンテキストがありません", "知りません", "答えが明確でありません", "すみません", 
      "答えがありません", "説明がありません", "リマインダー", "コンテキストが提供されていません", "有用な回答がありません", 
      "指定されたコンテキスト", "有用でありません", "適切ではありません", "質問がありません", "明確でありません",
      "十分な情報がありません", "適切な情報がありません", "直接関係しているものが無いようです"
      ]
    
    if answer is None: # 回答がNoneの場合は不正な回答
      results = False
    else: # badanswer phraseを含んでいる場合は不正な回答
      for phrase in badanswer_phrases:
        if phrase in answer.lower():
          result = False
          break
    
    return result


  def _get_answer(self, context, question, timeout_sec=60):

    '''' タイムアウトハンドリングありのLLMからの回答取得 '''

    # デフォルトの結果
    result = None

    # 終了時間の定義
    end_time = time.time() + timeout_sec

    # タイムアウトに対するトライ
    while time.time() < end_time:

      # レスポンス取得の試行
      try: 
        result = self.qa_chain.generate([{'context': context, 'question': question}])
        break # レスポンスが成功したらループをストップ

      # その他のエラーでも例外を発生
      except Exception as e:
        print(f'LLM QA Chain encountered unexpected error: {e}')
        raise e

    return result


  def get_answer(self, question):
    ''' 指定された質問の回答を取得 '''

    # デフォルトの結果
    result = {'answer':None, 'source':None, 'output_metadata':None}

    # 質問から一般的な略語を削除
    for abbreviation, full_text in self.abbreviations.items():
      pattern = re.compile(fr'\b({abbreviation}|{abbreviation.lower()})\b', re.IGNORECASE)
      question = pattern.sub(f"{abbreviation} ({full_text})", question)

    # 適切なドキュメントの取得
    docs = self.retriever.get_relevant_documents(question)

    # それぞれのドキュメントごとに ...
    for doc in docs:

      # ドキュメントのキー要素を取得
      text = doc.page_content
      source = doc.metadata['source']

      # LLMから回答を取得
      output = self._get_answer(text, question)
 
      # 結果からアウトプットを取得
      generation = output.generations[0][0]
      answer = generation.text
      output_metadata = output.llm_output

      # no_answer ではない場合には結果を構成
      if self._is_good_answer(answer):
        result['answer'] = answer
        result['source'] = source
        result['output_metadata'] = output_metadata
        break # 良い回答であればループをストップ
      
    return result

class MLflowQABot(mlflow.pyfunc.PythonModel):

    def __init__(self, llm, retriever):
        self.qabot = QABot(llm, retriever)
        
    def predict(self, inputs):
        questions = list(inputs['question'])

        # 回答の返却
        return [self.qabot.get_answer(q) for q in questions]
  
def _load_pyfunc(data_path):    
  device = 0 if torch.cuda.is_available() else -1
  tokenizer = AutoTokenizer.from_pretrained(data_path, use_fast=False)
  model = AutoModelForCausalLM.from_pretrained(data_path)
  pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, device=device
  )
  llm = HuggingFacePipeline(pipeline=pipe)

  # エンべディングにアクセスするためにベクトルストアをオープン
  embeddings = HuggingFaceEmbeddings(model_name='sonoisa/sentence-bert-base-ja-mean-tokens-v2')
  vector_store = FAISS.load_local(f"{data_path}/vector_db", embeddings)

  n_documents = 5 # 取得するドキュメントの数 
  retriever = vector_store.as_retriever(search_kwargs={'k': n_documents}) # 取得メカニズムの設定

  return MLflowQABot(llm, retriever)