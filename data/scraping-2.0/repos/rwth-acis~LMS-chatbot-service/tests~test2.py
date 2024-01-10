import logging
import sys
from langchain import OpenAI
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext
from llama_index.indices.knowledge_graph import GPTKnowledgeGraphIndex
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

#load data and build index
dbis_slides = SimpleDirectoryReader('../src/sorted_documents').load_data()
print(dbis_slides[0].text)

# NOTE: can take a while! 
graph_index = GPTKnowledgeGraphIndex.from_documents(
    dbis_slides, 
    max_triplets_per_chunk=2,
    service_context=service_context,
    include_embeddings=True
)
print(graph_index)

query_engine = graph_index.as_query_engine(
    include_text=False, 
    response_mode="tree_summarize",
    embdding_mode='hybrid',
    similarity_top_k=3
)

response = query_engine.query(
    "What is transactionmanagement?", 
)
print(response)
