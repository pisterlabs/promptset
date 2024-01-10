import os
_RePolyA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(_RePolyA)

from repolya._const import WORKSPACE_RAG
from repolya._log import logger_rag

from repolya.rag.vdb_faiss import (
    add_texts_to_faiss_OpenAI,
    get_faiss_OpenAI,
    delete_doc_from_faiss_OpenAI,
    show_faiss,
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain


_db_name = str(WORKSPACE_RAG / 'lj_rag_openai')

# _vdb = get_faiss_OpenAI(_db_name)
# def refresh_model(_vdb):
#     retriever = _vdb.as_retriever()
#     llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
#     model = RetrievalQAWithSourcesChain.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#     )
#     return model
# model = refresh_model(_vdb)
# question = "什么是等离子体垃圾处理系统？"
# response = model({"question": question}, return_only_outputs=True)
# print("Answer : ", response['answer'])
# print("Sources : ", response['sources'])

_texts = ['宋卓的电话是18622223333', '今天星期一', '明天星期二', '后天星期三', '大后天星期四']
_metadatas = [{'source': "tt"} for _ in range(len(_texts))]
_ids = add_texts_to_faiss_OpenAI(_texts, _metadatas, _db_name)
print(_ids)

_vdb = get_faiss_OpenAI(_db_name)
show_faiss(_vdb)

delete_doc_from_faiss_OpenAI(_db_name, 'tt')
_vdb = get_faiss_OpenAI(_db_name)
show_faiss(_vdb)

