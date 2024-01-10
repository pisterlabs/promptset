from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper, LLMPredictor, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.docstore import SimpleDocumentStore
from langchain.chat_models import ChatOpenAI
from dotenv.main import load_dotenv
import gradio
import sys
import os
import logging

load_dotenv()

# 记录日志
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

favorite_language = os.environ['LANGUAGE']
openai_api_key = os.environ['OPENAI_API_KEY']
docs_dir = os.environ['DOC_DIRS']

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 20
    chunk_size_limit = 600
    # 定义提示词
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    # 定义LLM模型
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs, api_key=openai_api_key))

    # 创建document
    documents = SimpleDirectoryReader(directory_path).load_data()
    # 创建parser解析document到node
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    # 创建docstore
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    # 创建服务化上下文，给后续的index使用
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    # 创建index
    index1 = GPTSimpleVectorIndex(nodes, docstore=docstore, service_context=service_context)
    # index2 = GPTListIndex(nodes, docstore=docstore, service_context=service_context)

    # index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    # index.save_to_disk('index.json')
    index1.save_to_disk('index1.json')
    # index2.save_to_disk('index2.json')
    return index1

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index1.json')
    print(input_text)
    response = index.query(input_text, response_mode="compact")
    print(response.response)
    return response.response

webui = gradio.Interface(fn=chatbot,
                     inputs=gradio.inputs.Textbox(lines=7, label="输入您的文本"),
                     outputs="text",
                     title="AI 知识库聊天机器人")

index = construct_index(docs_dir)
webui.launch(share=True)
