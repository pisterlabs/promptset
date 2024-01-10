import os
import time
import sysconfig
from modules import chat_options
from modules.chat_options import cmd_opts
from modules.chat_ui import create_ui,load_index
from env import ini_env
# patch PATH for cpm_kernels libcudart lookup
import sys
import os
import json


ini_env()

# 导入必要的库和模块
from llama_index import ServiceContext, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage
from langchain import OpenAI
from modules.chat_options import cmd_opts
from modules.context import Context

from llama_index.data_structs.node import NodeWithScore
from llama_index.response.schema import Response
from llama_index.utils import truncate_text
from llama_index import download_loader, GPTVectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from pathlib import Path
import os



# 初始化LLM预测器（这里使用gpt-3.5-turbo模型）
llm_predictor = LLMPredictor(llm=OpenAI(temperature=cmd_opts.temperature, model_name=cmd_opts.model_name))

# 构建服务上下文
service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            prompt_helper=PromptHelper(max_input_size=cmd_opts.max_input_size,
            max_chunk_overlap=cmd_opts.max_chunk_overlap,
            num_output=cmd_opts.num_output),
            chunk_size_limit=cmd_opts.chunk_size_limit
            )

# 构建存储上下文
storage_context = StorageContext.from_defaults(persist_dir=cmd_opts.persist_dir)

# 加载索引
index = load_index_from_storage(storage_context, service_context=service_context)
query_engine = index.as_query_engine(
    similarity_top_k=3,
     response_mode="simple_summarize"
)

def add_turn(turns, new_turn):
    turns.append(new_turn)
    if len(turns) > 5:
        del turns[0]

# 定义打印来源节点的函数
def pprint_source_node(
    source_node, source_length: int = 350, wrap_width: int = 70
) -> str:
    source_text_fmt = truncate_text(source_node.node.get_text().strip(), source_length)
    return "".join([
        f'(相似度{source_node.score}) ',  
        "\nnode id:",
        source_node.doc_id,  
        "\n",
        source_text_fmt]) 
def pprint_answer(response):
    # 初始化参考文档列表
    refDoc = []  
    
    # 遍历来源节点，获取参考文档
    for node in response.source_nodes:  
        if node.similarity is not None:  
            refDoc.append(pprint_source_node(node))
    
    # 根据是否需要显示引用，生成最终的回应
    
    res = "Agent0: "+"".join([
        response.response,  
        "\n引用:\n",  
        "\n".join(refDoc)])
    print(res)

turns=[]
while True:
    text_input = input("User: ")
    
    turns.append({"role":"user","content":text_input})
    turns_str = json.dumps(turns, ensure_ascii=False)
    print(turns_str)
    response = query_engine.query(turns_str)
    pprint_answer(response)
    add_turn(turns,{"role":"assistant","content":response.response}) # 只有这样迭代才能连续提问理解上下文

