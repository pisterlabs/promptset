from langchain_ChatGLM.chatglm_llm import ChatGLM
import os
import copy
import time
import requests
import json

# Global Parameters
VECTOR_SEARCH_TOP_K = 2
LLM_HISTORY_LEN = 5


def get_docs(question, url, top_k):
    data = {"query": question, "top_k": top_k}
    docs = requests.post(url, json=data)
    docs = json.loads(docs.content)
    docs = docs["docs"]
    return docs


def init_cfg(url_llm, LLM_HISTORY_LEN, V_SEARCH_TOP_K=2):
    global chatglm, embeddings, VECTOR_SEARCH_TOP_K
    VECTOR_SEARCH_TOP_K = V_SEARCH_TOP_K
    chatglm = ChatGLM(url=url_llm)
    chatglm.history_len = LLM_HISTORY_LEN


def get_knowledge_based_answer(query, history_obj, url_lucene):
    global chatglm

    if len(history_obj.history) > LLM_HISTORY_LEN:
        history_obj.history = history_obj.history[-LLM_HISTORY_LEN:]
    chatglm.history = history_obj.history

    new_list = [lst for lst in chatglm.history if lst[0] is not None]
    chatglm.history = new_list

    input_llm = f"""任务: 给一段对话和一个后续问题，将后续问题改写成一个独立的问题。确保问题是完整的，没有模糊的指代。
    ----------------
    聊天记录：
    {str(chatglm.history)}
    ----------------
    后续问题：{query}
    ----------------
    改写后的独立、完整的问题："""

    new_question = query

    if len(history_obj.history) > 0:
        new_question = chatglm(input_llm)

    yield_0 = "根据您之前的对话历史，对您的问题进行重构后变为：\n" + new_question + "\n"
    title0 = "第一步：问题重构"
    origin_content = title0 + ":\r" + yield_0
    for ch in origin_content:
        yield ch
        time.sleep(0.1)
    yield "\r\n==============================================================================================\r\n"

    docs = get_docs(new_question, url_lucene, VECTOR_SEARCH_TOP_K)
    yield_1 = "根据您的问题，检索出的相关文档有：\n"
    docs_string = ""
    for i, doc in enumerate(docs):
        yield_1 += f"第{i + 1}篇相关文档：\n{doc}\n"
        docs_string += f"第{i + 1}篇相关文档：\n{doc}\n"

    llm_input = f"""
    基于以下相关文档，尽可能专业地来回答用户的问题，不允许在答案中添加编造的成分。

相关文档:
{docs_string}

问题:
{new_question}
    """

    title1 = "第二步：相关文档检索"
    origin_content = title1 + ":\r" + yield_1
    for i in range(0, len(origin_content), 5):
        if i + 5 >= len(origin_content):
            yield origin_content[i:]
        else:
            yield origin_content[i: i + 5]
        time.sleep(0.05)
    yield "\r\n==============================================================================================\r\n"

    if len(chatglm.history):
        chatglm.history[-1][0] = new_question
    new_list = [lst for lst in chatglm.history if lst[0] is not None]

    history_obj.history = new_list
    result = chatglm(llm_input)
    yield_2 = "结合问题和相关文档，生成的最终的回答为：\n" + result
    title2 = "第三步：回复生成"
    origin_content = title2 + ":\r" + yield_2
    for ch in origin_content:
        yield ch
        time.sleep(0.1)
    yield "\r\n==============================================================================================\r\n"
