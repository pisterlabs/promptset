
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.generation.utils import GenerationConfig

from langchain.document_loaders import TextLoader
from langchain.embeddings import MiniMaxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import time
import torch
import streamlit as st
import json
from io import StringIO
import os
import arxiv
from count import count_everything


st.title("用Baichuan写arxiv最近的论文总结")

def arxiv_search(query):
    summary = {}
    search = arxiv.Search(
      query = query,
      max_results = 1,
      sort_by = arxiv.SortCriterion.SubmittedDate
    )
    
    for result in search.results():
        summary = {"title":result.title,"url":result.entry_id, "abstract":result.summary}
        filename = result.entry_id.split('/')[-1].replace('.','_')+".pdf"
        result.download_pdf(dirpath='./papers',filename=filename)

    return summary,filename


@st.cache_resource #缓存model，多次用相同参数调用function时，直接从cache返回结果，不重复执行function    
def load_model():

    ckpt_path = "/root/llm/Baichuan2-13B-Chat"
    # from_pretrained()函数中，device_map参数的意思是把模型权重按指定策略load到多张GPU中
    model = AutoModelForCausalLM.from_pretrained(ckpt_path,trust_remote_code=True,device_map="auto",torch_dtype=torch.float16)
    model.generation_config = GenerationConfig.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path,trust_remote_code=True)

    return model,tokenizer


def init_chat_history():

    with st.chat_message("assistant"):
        st.markdown("你好，我是论文分析助手")

    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    else:
        for msg in st.session_state.msgs:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

    return st.session_state.msgs


def clear_history():
    del st.session_state.msgs

def similarity_search_from_vectordb(query,db_name):
    serarch_results = db_name.similarity_search(query)

    prompt = []
    for result in serarch_results:
        prompt.append(result.page_content)               
        prompt.append("\n"+query)
        prompts = "".join(prompt)

    return prompts


# 获取路径下最新文件名
def get_latest_filename(dir):
    import os
    file_list = os.listdir(dir)
    file_list.sort(key=lambda fn: os.path.getmtime(dir + "/" + fn)
                      if not os.path.isdir(dir + "/" + fn) else 0)
    
    filepath = os.path.join(dir,file_list[-1])

    return filepath


def main():
    counter = count_everything()
    query = st.text_input("请用:red[英文]输入要查询的论文关键词")
    model, tokenizer = load_model()
    filename = ""
    try:    
        if st.button("arxiv论文快报"):

            summary,filename = arxiv_search(query)
            print(filename)
            abstract = summary["abstract"]
            template = f"""
                    你是一个研究员，为下面一段论文的摘要写总结,
                    {abstract},
                    总结：

                    \n\n
                """
            title = summary["title"]
            url = summary["url"]
            st.markdown("标题: "+title+"\n\n"+ "URL: "+url)
            messages = [{"role":"user","content":template}]
            start_time = time.perf_counter()
            placeholder = st.empty()
            for response in model.chat(tokenizer, messages, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    

            last_time = time.perf_counter()- start_time  # 时间单位s
            speed = counter.count_token(response,last_time)

            
            st.markdown(f":watermelon: :green[{speed}]")
            st.success("论文已下载")

        # if st.button("Embedding"):
        dir =  "/root/llm/baichuan-13B/langchain-llm/papers"
        paper_file = get_latest_filename(dir)

        # 把上传的文件导入到langchain的textloader
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(paper_file)
        pages = loader.load_and_split()

        # 把分割后到文本embedding成向量，嵌入到向量数据库
        db = FAISS.from_documents(pages,MiniMaxEmbeddings())
        st.success("文档转换为向量，成功！")

        msgs = init_chat_history()

        if query := st.chat_input("输入你的问题"):
            
            with st.chat_message("user"):
                st.markdown(query)
                prompts = similarity_search_from_vectordb(query,db)

            # with st.chat_message("assistant"):
            #     placeholder = st.empty()
            #     placeholder.markdown(prompts)

            template_query = f"""
            Answer the question based only on the content above:{query}
            """

            msgs.append({'role':'user','content':query})
            # msgs.append({'role':'user','content':prompts})        
            current_msgs = [{'role':'user','content':prompts+"\n\n"+template_query}]
            print(current_msgs)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                for responce in model.chat(tokenizer,current_msgs,stream=True):
                    placeholder.markdown(responce)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            msgs.append({'role':'assistant','content':responce})

            st.button("清空对话",on_click=clear_history)
            # print(json.dumps(st.session_state.msgs, ensure_ascii=False), flush=True)

    except KeyError:
        st.markdown("未找到相关论文")


    
if __name__ == "__main__":
    main()

