import os
import time
import sysconfig
from modules import chat_options
from modules.chat_options import cmd_opts
from env import ini_env

ini_env()

# 导入必要的库和模块
from llama_index import ServiceContext, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage
from langchain import OpenAI
from langchain.agents import Tool, initialize_agent
from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory



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


memory = GPTIndexChatMemory(
    index=index, 
    memory_key="chat_history", 
    query_kwargs={"response_mode": "simple_summarize"},
    # return_source returns source nodes instead of querying index
    return_source=True,
    # return_messages returns context in message format
    return_messages=True
)

tools = [
    Tool(
        name = "GPT Index",
        func=lambda q: str(index.as_query_engine( similarity_top_k=3, response_mode='tree_summarize',verbose=True).query(q)),
        description="useful for when you want to answer questions about 丰迈. The input to this tool should be a complete chinese sentence.",
        return_direct=True
    ),
]
llm=OpenAI(temperature=cmd_opts.temperature, model_name=cmd_opts.model_name)
agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory,verbose=True)

while True:
    text_input = input("User: ")
    response = agent_chain.run(input=text_input)
    print(f'Agent: {response}')