import os
_RePolyA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(_RePolyA)

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.document_transformers import (
    LongContextReorder,
    EmbeddingsRedundantFilter,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import BM25Retriever

from typing import List
from pydantic import BaseModel, Field

from repolya.local.textgen import get_textgen_llm
from repolya.rag.vdb_faiss import get_faiss_HuggingFace
from repolya.rag.qa_chain import pretty_print_docs
from repolya.rag.embedding import get_embedding_HuggingFace
from repolya.rag.vdb_faiss import show_faiss

from repolya._const import WORKSPACE_RAG


_textgen_url = "http://127.0.0.1:5552"


class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")
class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)
    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)
output_parser = LineListOutputParser()
QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""### 系统:

你是一名AI语言模型助手。
你的任务是生成五个给定用户问题的不同版本，用于从向量中检索相关文档数据库。
通过对用户问题产生多种观点，帮助用户克服基于距离的相似性搜索的一些限制。
请提供这些替代问题，并用换行符分隔。

### 操作说明:

原问题：{question}

### 回复:
""",
)

llm = get_textgen_llm(_textgen_url, _top_p=0.5, _max_tokens=200, _stopping_strings=["```", "###"])
llm_chain = LLMChain(
    llm=llm,
    prompt=QUERY_PROMPT,
    output_parser=output_parser,
)

_db_name = str(WORKSPACE_RAG / 'lj_rag_hf')
_vdb = get_faiss_HuggingFace(_db_name)
show_faiss(_vdb)
# exit()

_base_retriever = _vdb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, 'fetch_k': 20},
    # search_type="similarity_score_threshold",
    # search_kwargs={'score_threshold': 0.5},
)

_model_name, _embedding = get_embedding_HuggingFace()
_filter = EmbeddingsRedundantFilter(embeddings=_embedding)
_pipeline = DocumentCompressorPipeline(transformers=[_filter])
# ##### Re-order results to avoid performance degradation
# _reordering = LongContextReorder()
# _pipeline = DocumentCompressorPipeline(transformers=[_filter, _reordering])
# ##### ContextualCompressionRetriever
_compression_retriever_reordered = ContextualCompressionRetriever(
    base_compressor=_pipeline,
    base_retriever=_base_retriever
)

_multi_retriever = MultiQueryRetriever(
    retriever=_compression_retriever_reordered,
    llm_chain=llm_chain,
    parser_key="lines"
)

_query = "福特号舰长是谁？"

_run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
_generated_queries = _multi_retriever.generate_queries(_query, _run_manager)
print(_generated_queries)

_docs = _multi_retriever.get_relevant_documents(_query)
print("\n"+'='*100)
print(pretty_print_docs(_docs))



_bm25_retriever = BM25Retriever.from_documents(_docs)
_docs = _bm25_retriever.get_relevant_documents(_query)
print("\n"+'='*100)
print(pretty_print_docs(_docs))
# exit()


rag_prompt_yi = PromptTemplate(
    input_variables=["question", "context"],
    template="""### 系统:

您是问答任务的助手。使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三个句子并保持答案简洁通顺。

### 操作说明: 

上下文: 
{context} 

问题: {question}

### 回复:
""",
)

llm = get_textgen_llm(_textgen_url, _top_p=0.1, _max_tokens=2000, _stopping_strings=["```", "###", "\n\n"])
_qa = load_qa_chain(
    llm,
    chain_type='stuff',
    prompt=rag_prompt_yi
)
_ans = _qa(
    {
        "input_documents": _docs,
        "question": _query
    },
    return_only_outputs=True
)
print("\n"+'='*100)
print(_ans)
print(_ans['output_text'])

