import os
import time

from pytest import console_main
#key = os.environ["OPENAI_API_KEY"]
#searchkey = os.environ["SERPAPI_API_KEY"]

from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

from file import get_index_path, get_index_name_from_file_path, check_index_file_exists, \
    get_index_name_without_json_extension, clean_file, check_file_is_compressed, index_path, compress_path, \
    decompress_files_and_get_filepaths, clean_files, check_index_exists

from file import check_index_file_exists, get_index_filepath, get_name_with_json_extension

def get_persist_path_from_filename(filename):
    parent_dir = os.path.dirname(filename)
    persist_dir = os.path.join(parent_dir, os.path.basename(filename) + "_vector_store_dir")
    return persist_dir

def check_lang_file_exist(file_path):
    if os.path.exists(file_path):
        return True
    else:
        return False

def get_index_name_without_cache_extension(index_name):
    return index_name.replace(".cache", "")


def get_name_with_cache_extension(index_name):
    return index_name + '.cache'

def check_index_cache_exists(index_name):
    index_dir = get_name_with_cache_extension(index_name)
    if (os.path.exists(index_dir))is False:
        return False
    else:
        return True
    #persist_dir = get_name_with_cache_extension(index_name)
    #return check_lang_file_exist(index_name)

def create_lang_index(filepath):
    index_name = get_index_name_from_file_path(filepath)
    index = _create_index(filepath, index_name)
    return index_name, index

dict_chain={}
def _create_index(filepath, index_name):
    index = _load_index_by_index_name(index_name)
    if index is not None:
        return index

    persist_dir = get_name_with_cache_extension(index_name)
    
    start_time = time.time()        
    doc_name = os.path.basename(filepath)
    print(f"开始分析文档[{doc_name}]，时间：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    abs_file_path = os.path.abspath(filepath)    
    loader = filepath.endswith(".pdf") and PyPDFLoader(abs_file_path) or TextLoader(abs_file_path)
    # 定义文本分块的规则，这里用了一个很简单的规则，按照默认的分隔符来切割文本，使得每一段不超过1000个字符
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "，", "\n"])
    chunks = loader.load_and_split(splitter)

    # 将数据转成 document 对象，每个文件会作为一个 document
    # documents = loader.load()

    # 切割加载的 document
    split_docs = splitter.split_documents(chunks)

    # 初始化 openai 的 embeddings 对象
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
    docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_dir)
    docsearch.persist()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"分析文档[{doc_name}]执行时长：%02d:%02d:%02d" % (hours, minutes, seconds))
    
    return index

# 从磁盘加载后缀为 cache 的index文件加，如果不存在就返回 None
def _load_index_by_index_name(index_name):
    index_dir = get_name_with_cache_extension(index_name)
    if (os.path.exists(index_dir))is False:
        return None
    
    # 初始化 openai 的 embeddings 对象
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")    
    #index_filepath = get_index_filepath(index_dir)
    docsearch = Chroma(persist_directory=index_dir, embedding_function=embeddings)

    return docsearch

def get_answer_from_lang_index(question, index_name):
    chain2 = None
    if index_name in dict_chain:
        print(f'{index_name}:内存中已经存在，直接加载')
        chain2 = dict_chain[index_name]    
    else:        
        docsearch = _load_index_by_index_name(index_name)
        # 创建问答对象
        chain2 = get_chain(docsearch)
        dict_chain[index_name] = chain2 

    ret = chain2({"question": question})
    print("A:", ret['answer'])
    return ret['answer']

def test(file_path):
    # 获取持久化目录名
    persist_dir = get_persist_path_from_filename(file_path)
    
    # 初始化 openai 的 embeddings 对象
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # 加载数据
    if os.path.exists(persist_dir):
        docsearch = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        start_time = time.time()        
        doc_name = os.path.basename(file_path)
        print(f"开始分析文档[{doc_name}]，时间：", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
        loader = file_path.endswith(".pdf") and PyPDFLoader(file_path) or TextLoader(file_path)
        # 定义文本分块的规则，这里用了一个很简单的规则，按照默认的分隔符来切割文本，使得每一段不超过1000个字符
        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "，", "\n"])
        chunks = loader.load_and_split(splitter)

        # 将数据转成 document 对象，每个文件会作为一个 document
        # documents = loader.load()

        # 切割加载的 document
        split_docs = splitter.split_documents(chunks)

        # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
        docsearch = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_dir)
        docsearch.persist()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        print(f"分析文档[{doc_name}]执行时长：%02d:%02d:%02d" % (hours, minutes, seconds))
    
    # 创建问答对象
    chain2 = get_chain(docsearch)   

    # 进行问答
    # 下面就比较简单了，不断读取问题然后执行chain
    while True:
        question  = input("\nQ: ")
        if not question:
            break
        #print("A:", chain.run(question))    
        #ret = qa({"query": question})
        #ret = chain.run(question)
        #print("A:", ret)
        ret = chain2({"question": question})
        print("A:", ret['answer'])

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains import RetrievalQAWithSourcesChain
system_template="""Answer the user's question in Chinese using the following context snippets. 
If there is no relevant information mentioned in the document based on the following context, just say '文档中没有提到相关内容'.
If you don't know the answer, just say "嗯..., 我不知道答案.", don't try to make up an answer.
ALWAYS return a "Sources" part in your answer.
The "Sources" part should be a reference to the source of the document from which you got your answer.
Example of your response should be:
```
The answer is foo
Sources:
1. abc
2. xyz
```
Begin!
----------------
{summaries}
"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

def get_chain(store):
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0.2), 
        chain_type="stuff", 
        retriever=store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        reduce_k_below_max_tokens=True
    )
    return chain

#test('E:/tempdown/fz.pdf')
#test('E:/tempdown/xuexi.pdf')

def test_qa():
    #filename = 'E:/tempdown/documents/xuexi.pdf'
    filepath = os.path.join(get_index_path(), os.path.basename(filename))
    index_name, index = create_lang_index(filepath)
      
    while True:
        question  = input("\nQ: ")
        if not question:
            break
        #print("A:", chain.run(question))    
        #ret = qa({"query": question})
        #ret = chain.run(question)
        #print("A:", ret)
        answer = get_answer_from_lang_index(question, index_name)

        print("A:", answer)    

#test_qa()