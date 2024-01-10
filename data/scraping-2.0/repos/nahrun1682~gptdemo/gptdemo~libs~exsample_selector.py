import openai
import os

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
import pandas as pd
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.vectorstores import FAISS


print("ok")
# .envファイルの読み込み
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
#data/qaへの相対パス



#ファイルパス
test_excel_path ="gptdemo/data/qa/virusQA.xlsx"
DB_DIR = "tmp/ChromaDB"

#エクセルの読み込みからデータの変換を行う関数
def excel_to_qa_list(test_excel_path):
    df = pd.read_excel(test_excel_path)
    #データの変換
    qa_list = []
    for index, row in df.iterrows():
        input_value = row[1]  # 2列目の値を取得
        output_value = row[2]  # 3列目の値を取得
        qa_single = {"input": input_value, "output": output_value}  # 指定の形式に変換
        qa_list.append(qa_single)
    return qa_list

def edited_df_to_qa_list(edited_df):
    df = edited_df
    #データの変換
    qa_list = []
    for index, row in df.iterrows():
        input_value = row[0]  # 2列目の値を取得
        output_value = row[1]  # 3列目の値を取得
        qa_single = {"input": input_value, "output": output_value}  # 指定の形式に変換
        qa_list.append(qa_single)
    return qa_list


def get_qa(query,neighborNum):
    #定数設定
    load_dotenv()
    model_engine = os.environ["OPENAI_MODEL_ENGINE"]
    openai.api_key = os.environ["OPENAI_API_KEY"]

    #ファイル名から拡張子以外を取得
    filename = os.path.splitext(os.path.basename(test_excel_path))[0]

    #OpenAIAPIで埋め込みの用意
    embedding_openai = OpenAIEmbeddings()

    #プロンプトテンプレートの用意
    qa_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Q: {input}\n\nA: {output}",
    )

    #エクセルの読み込みからデータの変換実施
    qa_list = excel_to_qa_list(test_excel_path)
    
    qa_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    qa_list,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    embedding_openai,
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # This is the number of examples to produce.
    k=neighborNum
)
    similar_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=qa_selector,
    example_prompt=qa_prompt,
    prefix="■質問内容: {question}",
    suffix="質問内容に近い{}件のQAを表示しました".format(neighborNum),
    input_variables=["question"],
)

    return similar_prompt.format(question=query)

def get_qa_edited_df(query,neighborNum,edited_df):
    #定数設定
    load_dotenv()
    model_engine = os.environ["OPENAI_MODEL_ENGINE"]
    openai.api_key = os.environ["OPENAI_API_KEY"]

    #ファイル名から拡張子以外を取得
    # filename = os.path.splitext(os.path.basename(test_excel_path))[0]

    #OpenAIAPIで埋め込みの用意
    embedding_openai = OpenAIEmbeddings()

    #プロンプトテンプレートの用意
    qa_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Q: {input}\n\nA: {output}",
    )

    #エクセルの読み込みからデータの変換実施
    qa_list = edited_df_to_qa_list(edited_df)
    
    qa_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    qa_list,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    embedding_openai,
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # This is the number of examples to produce.
    k=neighborNum
)
    similar_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=qa_selector,
    example_prompt=qa_prompt,
    prefix="■質問内容: {question}",
    suffix="質問内容に近い{}件のQAを表示しました".format(neighborNum),
    input_variables=["question"],
)

    return similar_prompt.format(question=query)

    