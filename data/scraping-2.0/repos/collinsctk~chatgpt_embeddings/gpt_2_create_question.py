import numpy as np
import openai
import pandas as pd
import ast
from gpt_0_basic_info import api_key, csv_file_path
from gpt_1_embeddings_training import get_embedding
# 最大的内容长度
max_context_len = 1500

EMBEDDING_MODEL = "text-embedding-ada-002"

openai.api_key = api_key


def vector_similarity(x: list[float], y: list[float]):
    """
    计算并且返回两个向量的相似度
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return_np_dot_result = np.dot(np.array(x), np.array(y))
    return return_np_dot_result


def get_query_similarity(input_query: str, df: pd.DataFrame):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document
    embeddings to find the most relevant sections.
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    # 获取输入input_query的embedding向量
    query_embedding = get_embedding(input_query)

    # 算每一行的embedding向量和输入input_query的embedding向量的相似度
    df['similarities'] = df['embeddings'].apply(lambda x: vector_similarity(query_embedding, x))
    # print(df)
    """
                                           QandA  ... similarities
    0             当有人问：公司名称, 请回答：亁颐堂科技有限责任公司  ...     0.809908
    1        当有人问：亁颐堂是做什么的, 请回答：亁颐堂是一个网络培训公司  ...     0.877552
    2           当有人问：你们公司有多少人, 请回答：亁颐堂有三十多个人  ...     0.808605
    3  当有人问：你们公司有多少个分部, 请回答：亁颐堂有北京 上海和南京三个分部  ...     0.783896
    """

    # 找到最相似的两个
    two_largest = df['similarities'].nlargest(2).index.tolist()

    # print(two_largest)
    # [1, 0] 行的索引

    # 如果最相似的df.loc[two_largest[0]]['similarities']都小于0.8，那么就返回空字符串
    # 如果第二相似的df.loc[two_largest[1]]['similarities']小于0.8，并且拼接后长度大于1500，那么就返回df.loc[two_largest[0]]['QandA']
    # 如果第二个相似的df.loc[two_largest[1]]['similarities']大于0.8，那么就返回两个拼接后的字符串
    context = '' if df.loc[two_largest[0]]['similarities'] < 0.8 else df.loc[two_largest[0]]['QandA'] \
        if (df.loc[two_largest[1]]['similarities'] < 0.8 or (len(df.loc[two_largest[1]]['QandA'] + '\n' +
                                                                 df.loc[two_largest[0]]['QandA']) >= max_context_len)) \
        else (df.loc[two_largest[1]]['QandA'] + '\n' + df.loc[two_largest[0]]['QandA'])

    return context


def _decorate_query(input_query: str, df: pd.DataFrame) -> str:
    try:
        context = get_query_similarity(input_query, df)
        if context != '':
            header = """请使用上下文尽可能真实、自然地回答问题，如果答案未包含在上下文中，请不要编造回答，并且不要在回答中包含”根据上下文”这个短语。\n\n上下文：\n"""
            input_query = header + context + "\n\n 问题: " + input_query + "\n 回答:？"
        """
        请使用上下文尽可能真实、自然地回答问题，如果答案未包含在上下文中，请不要编造回答，并且不要在回答中包含”根据上下文”这个短语。

        上下文：
        当有人问：公司名称, 请回答：亁颐堂科技有限责任公司
        当有人问：亁颐堂是做什么的, 请回答：亁颐堂是一个网络培训公司
        
         问题: 亁颐堂是做什么的
         回答:？
        """
        return input_query
    except Exception as e:
        print(e)
        return input_query


def decorate_query(input_query: str, filepath) -> str:
    try:
        df = pd.read_csv(filepath)
        # 如果df为空，那么就返回input_query
        if df.empty:
            return input_query
        else:
            try:
                # 使用.apply()方法对'embeddings'列中的每个元素进行操作。
                # 用lambda函数定义了一个匿名函数，这个匿名函数接受一个参数x，并将ast.literal_eval(x)的结果返回。
                # ast.literal_eval(x)是Python中ast模块（Abstract Syntax Trees，抽象语法树）的literal_eval()函数，
                # 它安全地解析一个字符串形式的字面量表达式（如字符串形式的数字、列表、元组、字典等），并返回该表达式的对应Python对象。
                # 这里，它将字符串形式的x解析成一个Python对象。

                # print(df)
                """
                                                                   QandA                                         embeddings
                0             当有人问：公司名称, 请回答：亁颐堂科技有限责任公司  [-0.009181897155940533, -0.022621875628829002,...
                1        当有人问：亁颐堂是做什么的, 请回答：亁颐堂是一个网络培训公司  [-0.014206966385245323, -0.018791578710079193,...
                2           当有人问：你们公司有多少人, 请回答：亁颐堂有三十多个人  [-0.004695456940680742, -0.011140977963805199,...
                3  当有人问：你们公司有多少个分部, 请回答：亁颐堂有北京 上海和南京三个分部  [0.0038718082942068577, -0.003343536052852869,...
                """
                # df默认读出的embeddings是字符串，需要转换成list
                df['embeddings'] = df['embeddings'].apply(lambda x: ast.literal_eval(x))

                return _decorate_query(input_query, df)
            except Exception as e:
                print(e)
                return input_query
    except Exception as e:
        print(e)
        return input_query


if __name__ == '__main__':
    # query = '谁发现了牛顿三大定律'  # 不相关的就直接返回问题
    query = '亁颐堂是做什么的'   # 找到相关内容, 就添加上下文
    # 如果内容相关就添加如下上下文
    """
    请使用上下文尽可能真实、自然地回答问题，如果答案未包含在上下文中，请不要编造回答，并且不要在回答中包含”根据上下文”这个短语。

    上下文：
    当有人问：公司名称请回答：亁颐堂科技有限责任公司
    当有人问：亁颐堂是做什么的请回答：亁颐堂是一个网络培训公司

     问题: 亁颐堂是做什么的
     回答:？
    """
    print(decorate_query(query, filepath=csv_file_path))\



