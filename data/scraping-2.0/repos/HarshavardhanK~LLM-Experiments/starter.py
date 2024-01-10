from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import OpenAIEmbedding
from langchain.embeddings import OllamaEmbeddings
from llama_index import ServiceContext, set_global_service_context

from langchain.llms import Ollama


llm = Ollama(model="llama2")

embed_model = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")

embed_model_open_ai = OpenAIEmbedding()

service_context = ServiceContext.from_defaults(embed_model=embed_model_open_ai, llm=llm)

# optionally set a global service context
set_global_service_context(service_context)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(service_context=service_context)
#response = query_engine.query("What did the shopkeeper ask Harsha in Manali? Can you answer the shopkeeper's question?")
response = query_engine.query("What is The rational agent approach?")
print(response)

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#With Llama2 Embedding:
'''In Kirchenbauer's paper on "A Watermarking Algorithm for Large Language Models," the parameters of gamma and delta play a crucial role in the proposed watermarking algorithm.

Gamma (γ) represents the step size or stretching factor, which determines how much the watermark is stretched or compressed during embedding. A larger value of γ results in a stronger watermark, but it also increases the risk of detection by an attacker. On the other hand, a smaller value of γ leads to a weaker watermark, but it reduces the risk of detection.

Delta (δ) represents the threshold or decision point, which determines the level of similarity between the embedded watermark and the cover text required for detection. A larger value of δ results in a higher detection probability, but it also increases the risk of false positives. On the other hand, a smaller value of δ leads to a lower detection probability, but it reduces the risk of false positives.'''