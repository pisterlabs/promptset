
import openai, os
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import SpacyTextSplitter
from llama_index import GPTListIndex, LLMPredictor, ServiceContext, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

openai.api_key = os.environ.get("OPENAI_API_KEY")

# define LLM, 设置了模型输出的内容都在 1024 个 Token 以内，这样可以确保我们的小结不会太长，不会把一大段不相关的内容都合并到一起去。
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1024))

# 使用 SpacyTextSplitter 来进行中文文本的分割, 我们也限制了分割出来的文本段，最长不要超过 2048 个 Token, 这些参数都可以根据你实际用来处理的文章内容和属性自己设置
text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size = 2048)
parser = SimpleNodeParser(text_splitter=text_splitter)
documents = SimpleDirectoryReader('./articles').load_data()
nodes = parser.get_nodes_from_documents(documents)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# GPTListIndex 在构建索引的时候，并不会创建 Embedding，所以索引创建的时候很快，也不消耗 Token 数量。它只是根据你设置的索引结构和分割方式，建立了一个 List 的索引。
list_index = GPTListIndex(nodes=nodes, service_context=service_context)

response = list_index.query("下面鲁迅先生以第一人称‘我’写的内容，请你用中文总结一下:", response_mode="tree_summarize")
print(response)