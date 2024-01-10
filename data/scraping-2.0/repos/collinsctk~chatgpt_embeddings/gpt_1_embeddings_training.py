# 参考文章
# https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb
import openai
import pandas as pd
from gpt_0_basic_info import api_key, excel_file_path, csv_file_path

EMBEDDING_MODEL = "text-embedding-ada-002"

openai.api_key = api_key

"""
We preprocess the document sections by creating an embedding vector for each section. An embedding is a vector of 
numbers that helps us understand how semantically similar or different the texts are. The closer two embeddings are to 
each other, the more similar are their contents.

翻译:
我们通过为每个部分创建嵌入向量来预处理文档部分。嵌入是一组数字，帮助我们理解文本的语义相似性或差异。两个嵌入越接近，它们的内容就越相似。
"""


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    # 计算嵌入向量
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return_data_embedding = result["data"][0]["embedding"]
    # 具体数据如下
    # [-0.008970350958406925, -0.014719498343765736, ~~~~很多很多~~~~]
    return return_data_embedding


def compute_doc_embeddings(df: pd.DataFrame):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dataframe with embedding
    """
    # print(df)
    """
                    prompt  ...                                  QandA
    0        当有人问：公司名称  ...             当有人问：公司名称, 请回答：亁颐堂科技有限责任公司
    1    当有人问：亁颐堂是做什么的  ...        当有人问：亁颐堂是做什么的, 请回答：亁颐堂是一个网络培训公司
    2    当有人问：你们公司有多少人  ...           当有人问：你们公司有多少人, 请回答：亁颐堂有三十多个人
    3  当有人问：你们公司有多少个分部  ...  当有人问：你们公司有多少个分部, 请回答：亁颐堂有北京 上海和南京三个分部
    """
    # 添加新的列'embeddings', 值为'QandA'这个列计算的向量数据
    df['embeddings'] = df['QandA'].apply(lambda x: get_embedding(x))
    return df


def get_embeddings(excel_file_path, csv_file_path):
    df = pd.read_excel(excel_file_path)
    # 删除换行"\n"
    df['prompt'] = df['prompt'].apply(lambda x: x.replace('\n', ''))
    # 给问题加上"当有人问："的前缀
    df['prompt'] = df['prompt'].apply(lambda x: "当有人问：" + x + '')
    # 给答案加上", 请回答："的前缀
    df['completion'] = df['completion'].apply(lambda x: ", 请回答：" + x)
    # 将问题和答案合并
    df['QandA'] = df['prompt'] + df['completion']
    # 只取'QandA'和'embeddings'两列
    df = compute_doc_embeddings(df)[['QandA', 'embeddings']]
    # print(df)
    """
                                           QandA                                         embeddings
    0             当有人问：公司名称, 请回答：亁颐堂科技有限责任公司  [-0.009215764701366425, -0.022858258336782455,...
    1        当有人问：亁颐堂是做什么的, 请回答：亁颐堂是一个网络培训公司  [-0.014166537672281265, -0.01877765916287899, ...
    2           当有人问：你们公司有多少人, 请回答：亁颐堂有三十多个人  [-0.004638118669390678, -0.011072063818573952,...
    3  当有人问：你们公司有多少个分部, 请回答：亁颐堂有北京 上海和南京三个分部  [0.0038256149273365736, -0.0033990885131061077...
    """
    df.to_csv(csv_file_path, index=False, encoding='utf-8_sig')


if __name__ == '__main__':
    get_embeddings(excel_file_path, csv_file_path)
