from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, GPTListIndex, GPTKeywordTableIndex, \
    GPTKnowledgeGraphIndex
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index import StorageContext, ServiceContext, LLMPredictor
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.graph_stores import SimpleGraphStore
from llama_index.vector_stores import SimpleVectorStore
from langchain.chat_models import ChatOpenAI
import networkx as nx
import matplotlib.pyplot as plt
from llama_index import load_index_from_storage
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter

PERSIST_DIR = '/home/przemek/deep-learning/road-to-deep-learning/road-to-deep-learning/RAG/llama_index_/tutorials/pdf/index_stored'


llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager, llm_predictor=llm_predictor)

'''
GPTVectorStoreIndex - creates numerical vectors from the text using word embeddings and retrieves relevant documents 
based on the similarity of the vectors
Advantage
- It is cheap to index and retrieve the data

Disadvantage:
- the quality of the answers depends on the quality of the embedding

Tip:
- skipping the table of contents when you index the documents   
'''

documents = SimpleDirectoryReader("/home/przemek/GAT/knowledge-base/vault", recursive=True).load_data()
text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)

node_parser = SimpleNodeParser(text_splitter=text_splitter)
nodes = node_parser.get_nodes_from_documents(documents)
for node in nodes:
    node.metadata['link'] = 'metadata_dict[node.metadata]'
index = GPTVectorStoreIndex(nodes, service_context=service_context)

# persist index in local folder
index.storage_context.persist(persist_dir=PERSIST_DIR)

# load index
# storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
# loaded_index = load_index_from_storage(storage_context)


'''
GPTListIndex - perfect when you don’t have many documents. index concatenates all chunks and sends them all to the LLM. 
If the resulting text is too long, the index splits the text and asks LLM to refine the answer
Entire work happens when we send the question, not when we index the data.
Advantage
- few questions to answer using a handful of documents
- give us the best answer because AI will get all the available data

Disadvantage:
- expensive
'''

# index = GPTListIndex.from_documents(documents, service_context=service_context)


'''
GPTKeywordTableIndex - extracts the keywords from indexed nodes and uses them to find relevant documents.
When we ask a question, first, the implementation will generate keywords from the question.
Next, the index searches for the relevant documents and sends them to the LLM.
The bulk of the work happens at the indexing time.

Watch out: Every node is sent to the LLM to generate keywords. Sending every document to an LLM skyrockets the 
cost of indexing

Disadvantages:
- slower and more expensive than the indexes we have seen so far
'''

# index = GPTKeywordTableIndex.from_documents(documents, service_context=service_context)


'''
GPTKnowledgeGraphIndex - building a knowledge graph requires even more resources.  
Builds a knowledge graph with keywords and relations between them.
The retrieval part is similar to the keyword-based index. First, we get the keywords from the question. After that, we 
search for the relevant nodes using the knowledge graph and pass the documents to the LLM to generate the answer.

Advantage:
- We can retrieve the knowledge graph, and we can even visualize it

'''

# index = GPTKnowledgeGraphIndex.from_documents(documents, service_context=service_context)
# fig = plt.figure(1, figsize=(100, 40), dpi=30)
# nx.draw_networkx(index.get_networkx_graph(), font_size=18)

'''
Query
'''
query_engine = index.as_chat_engine(streaming=True)
response = query_engine.stream_chat("Why are Drive Events can be not available?")
print(response)
