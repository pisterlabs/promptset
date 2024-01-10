from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import SpacyTextSplitter
from llama_index import GPTListIndex, LLMPredictor, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, GPTSimpleKeywordTableIndex


# define LLM
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1024))
text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size = 2048)
parser = SimpleNodeParser(text_splitter = text_splitter)
documents = SimpleDirectoryReader('/Users/zm/aigcData/index_mr_fujino').load_data()
nodes = parser.get_nodes_from_documents(documents)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

list_index = GPTListIndex(nodes=nodes, service_context=service_context)

response = list_index.as_query_engine().query("下面鲁迅先生以第一人称'我'写的内容，请你用中文总结一下：")
print(response)