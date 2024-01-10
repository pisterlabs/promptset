# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: doc_qa.py
# @time: 2023/7/22 14:06
import cohere

from utils.db_client import es_client, milvus_client
from data_process.data_processor import get_text_embedding
from common.llm_chat_api import chat_completion
from utils.logger import logger
from config.config_parser import MILVUS_SIZE, ES_SIZE, MILVUS_THRESHOLD, EMBEDDING_MODEL, COHERE_API_KEY, RERANK_TOP_N


# 文档问答
class DocQA(object):
    def __init__(self, query):
        self.query = query

    def get_milvus_search_result(self):
        # milvus search content
        vectors_to_search = [get_text_embedding(self.query, EMBEDDING_MODEL)]
        # 通过嵌入向量相似度获取相似文本
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10},
        }
        result = milvus_client.search(vectors_to_search,
                                      "embeddings",
                                      search_params,
                                      limit=MILVUS_SIZE,
                                      output_fields=["text", "source"])
        # filter by similarity score
        return [(_.entity.get('text'), _.entity.get('source')) for _, dist in zip(result[0], result[0].distances) if dist > MILVUS_THRESHOLD]

    def get_es_search_result(self):
        result = []
        # 查询数据(全文搜索)
        dsl = {
            'query': {
                'match': {
                    'content': self.query
                }
            },
            "size": ES_SIZE
        }
        search_result = es_client.search(index='docs', body=dsl)
        if search_result['hits']['hits']:
            result = [(_['_source']['content'], _['_source']['source']) for _ in search_result['hits']['hits']]
        return result

    def get_context(self):
        contents = []
        # 去重
        milvus_search_result = self.get_milvus_search_result()
        es_search_result = self.get_es_search_result()
        for content_source_tuple in milvus_search_result + es_search_result:
            content, source = content_source_tuple
            if [content, source] not in contents:
                contents.append([content, source])
        return contents

    def rerank(self):
        before_rerank_contents = self.get_context()

        cohere_client = cohere.Client(COHERE_API_KEY)
        docs, sources = [_[0] for _ in before_rerank_contents], [_[1] for _ in before_rerank_contents]
        results = cohere_client.rerank(model="rerank-multilingual-v2.0",
                                       query=self.query,
                                       documents=docs,
                                       top_n=RERANK_TOP_N)

        after_rerank_contents = []
        for hit in results:
            after_rerank_contents.append([hit.document['text'], sources[hit.index]])
            logger.info(f"score: {hit.relevance_score}, query: {self.query}, text: {hit.document['text']}")
        return after_rerank_contents

    def get_qa_prompt(self):
        # 建立prompt
        prefix = "<文本片段>:\n\n"
        suffix = f"\n<问题>: {self.query}\n<回答>: "
        prompt = []
        contexts = []
        sources = []
        for i, text_source_tuple in enumerate(self.rerank()):
            text, source = text_source_tuple
            prompt.append(f"{i+1}: {text}\n")
            contexts.append(f"<{i+1}>: {text}")
            sources.append(f"<{i+1}>: {source}")
        qa_chain_prompt = prefix + ''.join(prompt) + suffix
        contexts, sources = "\n\n".join(contexts), "\n\n".join(sources)
        logger.info(qa_chain_prompt)
        return qa_chain_prompt, contexts, sources

    def answer(self, model_name):
        message, contexts, sources = self.get_qa_prompt()
        result = chat_completion(message, model_name)
        return result, contexts, sources


if __name__ == '__main__':
    test_model_name = "Baichuan-13B-Chat"
    # question = '美国人什么时候登上月球的？'
    # question = '戚发轫的职务是什么？'
    # question = '你知道格里芬的职务吗？'
    question = '格里芬发表演说时讲了什么？'
    # question = '日本的面积有多大？'
    reply = DocQA(question).answer(model_name=test_model_name)
    print(reply)

