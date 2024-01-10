# Hacky script made by langchain newb with https://www.youtube.com/watch?v=ypzmPwLH_Q4 as a base
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
import pinecone
import time
from llama_cpp import Llama

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate



device = 'mps' if torch.backends.mps.is_available() else 'cpu'
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs = {"device": device},
    encode_kwargs={"device": device, "batch_size": 32}
)

docs = [
    "Amadeus is an AI modelled after the researcher Makise Kurisu.",
    "Microwaving Bananas can lead to time travel"
]

embeddings = embed_model.embed_documents(docs)
print("Embeddings | Dimensionality")
print(len(embeddings),len(embeddings[0]))

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('./data/amadeus_info.md').load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, embed_model)

query = "What is Amadeus?"
docs = db.similarity_search(query)
print(docs[0].page_content)


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="./augmental-unholy-13b-q5km.gguf",
    temperature=0.75,
    max_tokens=3000,
    n_ctx=4096, 
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

prompt = """What is Amadeus?"""
# llm(prompt)

rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff',retriever=db.as_retriever(search_kwargs={"k":1})
)

rag_pipeline(prompt)





# # get API key from app.pinecone.io and environment from console
# pinecone.init(
#     api_key=os.environ.get('PINECONE_API_KEY') or '78c84b68-631e-46e8-a6e1-a963b734d384',
#     environment=os.environ.get('PINECONE_ENVIRONMENT') or 'gcp-starter'
# )

# index_name = 'llama-2-rag-test'

# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(
#         index_name,
#         dimension=len(embeddings[0]),
#         metric='cosine'
#     )
#     # wait for index to finish initialization
#     while not pinecone.describe_index(index_name).status['ready']:
#         time.sleep(1)
        
# index = pinecone.Index(index_name)
# index.describe_index_stats()

# from datasets import load_dataset

# data = load_dataset(
#     'jamescalam/llama-2-arxiv-papers-chunked',
#     split='train'
# )

# print(data)

# data = data.to_pandas()

# batch_size = 32
# from tqdm import tqdm

# for i in tqdm(range(0, len(data), batch_size)):
#     i_end = min(len(data), i+batch_size)
#     batch = data.iloc[i:i_end]
#     ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
#     texts = [x['chunk'] for i, x in batch.iterrows()]
#     embeds = embed_model.embed_documents(texts)
#     # get metadata to store in Pinecone
#     metadata = [
#         {'text': x['chunk'],
#          'source': x['source'],
#          'title': x['title']} for i, x in batch.iterrows()
#     ]
#     # add to Pinecone
#     index.upsert(vectors=zip(ids, embeds, metadata))

# index.describe_index_stats()

# m = Llama(model_path=MODEL,
#                  offload_kqv=True,
#                  n_ctx=3900,
#                  n_gpu_layers=1000)