import os
import openai
import tiktoken
from dotenv import load_dotenv
from llama_index.llms import OpenAI
from llama_index import set_global_service_context
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext, LLMPredictor, OpenAIEmbedding, PromptHelper


load_dotenv(dotenv_path="../.env")
openai_api_key = os.getenv("openai_key")

documents = SimpleDirectoryReader(input_dir='data').load_data()
llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)

embed_model = OpenAIEmbedding()

prompt_helper = PromptHelper(context_window=4096, num_output=256, chunk_overlap_ratio=0.1, chunk_size_limit=None)
text_splitter = TokenTextSplitter(
  separator=" ",
  chunk_size=1024,
  chunk_overlap=20,
  backup_separators=["\n"],
  tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

node_parser = SimpleNodeParser.from_defaults()


service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser, prompt_helper=prompt_helper)

index = VectorStoreIndex.from_documents(documents, service_context = service_context)
index.storage_context.persist()

query_engine = index.as_query_engine(service_context=service_context)
response = query_engine.query("What is HNSW?")
print(response)
