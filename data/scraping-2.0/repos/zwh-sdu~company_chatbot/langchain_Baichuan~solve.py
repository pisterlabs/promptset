from langchain_Baichuan.baichuan_llm import Baichuan
import requests
import json

# Global Parameters
RETRIEVAL_TOP_K = 2
LLM_HISTORY_LEN = 16


def init_cfg(url_llm):
    global llm, RETRIEVAL_TOP_K
    llm = Baichuan(url=url_llm)


def get_docs(question: str, url: str, top_k=LLM_HISTORY_LEN):
    data = {"query": question, "top_k": top_k}
    docs = requests.post(url, json=data)
    docs = json.loads(docs.content)
    docs = docs["docs"]
    return docs


def get_knowledge_based_answer(query, history_obj, url_lucene):
    global llm, RETRIEVAL_TOP_K

    if len(history_obj.history) > LLM_HISTORY_LEN:
        history_obj.history = history_obj.history[-LLM_HISTORY_LEN:]

    # 获取相关文档
    docs = get_docs(query, url_lucene, RETRIEVAL_TOP_K)
    doc_string = ""
    for i, doc in enumerate(docs):
        doc_string = doc_string + "第" + str(i + 1) + "段参考资料：" + doc + "\n"
    history_obj.history.append(
        {
            "role": "user",
            "content": f"""请根据以下参考资料中的信息全面具体地回答问题，不要尝试用你自己已有的知识回答。
---------
参考资料：
{doc_string}
---------
问题：
"""
            + query,
        }
    )

    # 调用大模型获取回复
    response = llm(history_obj.history)

    # 修改history，将之前的参考资料从history删除，避免history太长
    history_obj.history[-1] = {"role": "user", "content": query}
    history_obj.history.append({"role": "assistant", "content": response})
    if len(history_obj.history) > LLM_HISTORY_LEN:
        del history_obj.history[-LLM_HISTORY_LEN:]

    with open("./history.txt", "w") as file:
        file.write(doc_string)
        for item in history_obj.history:
            file.write(str(item) + "\n")
    yield response
