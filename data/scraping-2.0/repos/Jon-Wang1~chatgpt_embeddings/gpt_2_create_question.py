import openai
from gpt_0_basic_info import api_key
from gpt_0_create_qdrant import qdrant_url, collection_name
from gpt_1_embeddings_training import get_embedding
from qdrant_client import QdrantClient

client = QdrantClient(qdrant_url)

# 最大的内容长度
max_context_len = 1500

EMBEDDING_MODEL = "text-embedding-ada-002"

openai.api_key = api_key


def get_query_similarity(input_query):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document
    embeddings to find the most relevant sections.
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    # 获取输入input_query的embedding向量
    query_vector = get_embedding(input_query)

    # 从Qdrant中搜索与query_vector最相似的两个向量
    search_results = client.search(collection_name, query_vector, limit=2)

    # 找到vector最接近的两个QandA
    two_largest = []

    for result in search_results:
        two_largest.append({'similarities': result.score, 'QandA': result.payload.get('QandA')})

    # print(two_largest)
    # [{'similarities': 0.87828124, 'QandA': '当有人问：亁颐堂是做什么的, 请回答：亁颐堂是一个网络培训公司'},
    # {'similarities': 0.812168, 'QandA': '当有人问：公司名称, 请回答：亁颐堂科技有限责任公司'}]

    # 如果最相似的two_largest[0]['similarities']都小于0.8，那么就返回空字符串
    # 如果第二相似的two_largest[1]['similarities']小于0.8，并且拼接后长度大于1500，那么就返回two_largest[0]['QandA']
    # 如果第二个相似的two_largest[1]['similarities']大于0.8，那么就返回两个拼接后的字符串
    context = '' if two_largest[0]['similarities'] < 0.8 else two_largest[0]['QandA'] \
        if (two_largest[1]['similarities'] < 0.8 or (len(two_largest[1]['QandA'] + '\n' + two_largest[0]['QandA']) >= max_context_len)) \
        else (two_largest[1]['QandA'] + '\n' + two_largest[0]['QandA'])

    return context


def decorate_query(input_query: str) -> str:
    try:
        context = get_query_similarity(input_query)
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
        # print(e)
        return input_query


if __name__ == '__main__':
    # query = '谁发现了牛顿三大定律'  # 不相关的就直接返回问题
    # query = '亁颐堂是做什么的'   # 找到相关内容, 就添加上下文
    query = '现任明教教主是谁'  # 不相关的就直接返回问题
    """
    请使用上下文尽可能真实、自然地回答问题，如果答案未包含在上下文中，请不要编造回答，并且不要在回答中包含”根据上下文”这个短语。

    上下文：
    当有人问：公司名称请回答：亁颐堂科技有限责任公司
    当有人问：亁颐堂是做什么的请回答：亁颐堂是一个网络培训公司

     问题: 亁颐堂是做什么的
     回答:？
    """
    print(decorate_query(query))


