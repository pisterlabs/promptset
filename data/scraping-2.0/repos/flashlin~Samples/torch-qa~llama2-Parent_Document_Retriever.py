#!pip -q install langchain openai tiktoken chromadb lark
# !pip -q install sentence_transformers
# !pip -q install -U FlagEmbedding

from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
## Text Splitting & Docloader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader
## LLM
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp



loaders = [
    TextLoader('./data/kcad2.md'),
    TextLoader('./data/vue3.txt'),
]
docs = []
for l in loaders:
    docs.extend(l.load())


parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)


model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)
vectorstore = Chroma(collection_name="split_parents", embedding_function=bge_embeddings)
store = InMemoryStore()

big_chunks_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
big_chunks_retriever.add_documents(docs)

## Load LLM
llm = LlamaCpp(
        model_path='./models/Llama-2-7b-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf',
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        n_ctx=2048,  # 請求上下文 ValueError: Requested tokens (1130) exceed context window of 512
        # callback_manager=callback_manager,
        verbose=False,  # True
        streaming=True,
    )
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=big_chunks_retriever)

print("starting")
query = "How to debug vitest test files in VSCode?"
answer = qa.run(query)
print(f"{answer=}")
