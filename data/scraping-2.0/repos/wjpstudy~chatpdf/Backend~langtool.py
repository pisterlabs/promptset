import os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext
from llama_index import ServiceContext, LLMPredictor, PromptHelper,QuestionAnswerPrompt
from llama_index.indices.query.schema import QueryBundle
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer
from langchain  import  OpenAI

from utils import get_static_file_path

os.environ['OPENAI_API_KEY'] = 'Your Key'
os.environ['OPENAI_API_BASE'] = 'https://OpenAI Or Proxy'

max_tokens=1024
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", streaming=True, max_tokens=max_tokens))

indexInMemory = {}

QA_PROMPT_TMPL = (
    "We have provided context information below.\n"
    "---\n"
    "{context_str}"
    "\n---\n"
    "Given this information: {query_str}\n"
)

QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

def pdf_reader(file_path: str):
    """根据文件路径加载Document对象"""
    return SimpleDirectoryReader(input_files=[file_path]).load_data()

def build_index(document):
    """根据Document对象创建索引"""
    return GPTVectorStoreIndex.from_documents(document)

def save_index_to_storage(index, path):
    """保存索引文件到硬盘"""
    index.set_index_id(path)
    index.storage_context.persist(path)


def load_index_from_storage2(sign):
    """从硬盘加载索引文件"""
    path = get_static_file_path("index", sign)
    storage_context = StorageContext.from_defaults(persist_dir=path)
    return load_index_from_storage(storage_context, index_id=path)

def save_index_to_memory(sign, index):
    """保存索引到内存"""
    indexInMemory[sign] = index

def load_index_from_memory(sign):
    """从内存加载索引, 如果没有则从硬盘加载"""
    index = indexInMemory.get(sign)
    if index is None:
        index = load_index_from_storage2(sign)
        save_index_to_memory(sign, index)
    return index


def ask_query(index, text):

    max_input_size = 4096
    # set number of output tokens
    num_outputs = 1024
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600
    # index = GPTListIndex.from_documents(documents)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    query_engine = index.as_query_engine(
        #response_mode="tree_summarize",
        service_context=service_context,
        optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.5),
        streaming=True,
        prompt_helper=PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit),
        similarity_top_k=3,
        text_qa_template=QA_PROMPT
    )
    response = query_engine.query(text)
    return response.get_formatted_sources(length=600), response.response_gen


def index_search(sign, text):
    """索引搜索"""
    index = load_index_from_memory(sign)
    response = index.as_retriever().retrieve(QueryBundle(text))
    return response