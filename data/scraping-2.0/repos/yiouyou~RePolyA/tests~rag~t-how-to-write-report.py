import os
_RePolyA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(_RePolyA)

from repolya._const import WORKSPACE_RAG, AUTOGEN_JD
from repolya._log import logger_rag, logger_yj

from repolya.rag.doc_loader import clean_txt
from repolya.toolset.tool_langchain import (
    bing,
    ddg,
    google,
)
from repolya.rag.qa_chain import qa_vdb_multi_query
from repolya.rag.vdb_faiss import get_faiss_OpenAI

from langchain.document_loaders import WebBaseLoader
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQAWithSourcesChain

import re

from repolya.autogen.wf_jd import generate_context_for_each_query


def clean_filename(text, max_length=10):
    # 移除非法文件名字符（例如: \ / : * ? " < > |）
    _clean = re.sub(r'[\\/*?:"<>|]', '', text)
    # 替换操作系统敏感的字符
    _clean = _clean.replace(' ', '_')  # 替换空格为下划线
    # 取前 max_length 个字符作为文件名
    return _clean[:max_length]


_query = "2023年台风杜苏芮 AND 发生发展的时间线"
_event = "2023年台风杜苏芮"
_event_name = clean_filename(_event, 20)
_event_dir = str(AUTOGEN_JD / _event_name)
_key = _query.split(" AND ")[1]
_db_name = os.path.join(_event_dir, f"{_key}/yj_rag_openai")
_clean_txt_dir = os.path.join(_event_dir, f"{_key}/yj_rag_clean_txt")
_context, _token_cost = generate_context_for_each_query(_query, _db_name, _clean_txt_dir)
print(_context)
print(_token_cost)


def search_all(_query):
    _all = []
    _all.extend(bing(_query, n=1))
    _all.extend(ddg(_query, n=1))
    _all.extend(google(_query, n=1))
    return _all

def print_search_all(_all):
    _str = []
    for i in _all:
        _str.append(f"{i['link']}\n{i['title']}")
        # _str.append(f"{i['link']}\n{i['title']}\n{i['snippet']}")
    return "\n" + "\n".join(_str)

def clean_filename(text, max_length=10):
    # 移除非法文件名字符（例如: \ / : * ? " < > |）
    _clean = re.sub(r'[\\/*?:"<>|]', '', text)
    # 替换操作系统敏感的字符
    _clean = _clean.replace(' ', '_')  # 替换空格为下划线
    # 取前 max_length 个字符作为文件名
    return _clean[:max_length]

def fetch_all_link(_all, _event_dir):
    _txt_fp = []
    _all_link = [i['link'] for i in _all]
    _all_title = [i['title'] for i in _all]
    # print(_all_link)
    loader = WebBaseLoader()
    _re = loader.scrape_all(_all_link)
    for i in range(len(_re)):
        _fn = clean_filename(_all_title[i])
        _fp = os.path.join(_event_dir, f"{_fn}.txt")
        with open(_fp, "w") as wf:
            _txt = _re[i].get_text()
            # _txt = clean_txt(_re[i].get_text())
            wf.write(_txt)
        logger_yj.info(f"{_all_link[i]} -> {_fn}.txt")
        _txt_fp.append(_fp)
    return _txt_fp

def search_fetch(_query):
    _all = search_all(_query)
    print(print_search_all(_all))
    fetch_all_link(_all, "event_dir")

# _re = search_fetch(_query)
# print(_re)



# _event_name = "台风杜苏芮对福建的影响_"
# _db_name = str(AUTOGEN_JD / _event_name / f"yj_rag_openai")
# _vdb = get_faiss_OpenAI(_db_name)
# i_ans, i_step, i_token_cost = qa_vdb_multi_query(_query, _vdb, 'stuff')
# print(i_ans)
# print(i_step)
# print(i_token_cost)



def ask_vdb_with_source(_ask, _vdb):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vdb.as_retriever(),
    )
    with get_openai_callback() as cb:
        _res = chain({"question": _ask}, return_only_outputs=True)
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        logger_yj.info(_token_cost)
    return _res['answer'], _res['sources'], _token_cost

# i_ans, i_source, i_token_cost = ask_vdb_with_source(_query, _vdb)
# i_qas = f"Q: {_query}\nA: {i_ans.strip()}\nSource: {i_source}"
# print(i_qas)

