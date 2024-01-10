import sys
sys.path.insert(0, "/Users/slc/code/apecloud-gpt/llama_index/")
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from llama_index.indices.loading import load_index_from_storage
from llama_index import LLMPredictor
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index import (
    GPTVectorStoreIndex,
    LangchainEmbedding,
    ServiceContext,
    StorageContext,
    download_loader,
    PromptHelper
)


PyMuPDFReader = download_loader("PyMuPDFReader")

documents = PyMuPDFReader().load(file_path='./IPCC_AR6_WGII_Chapter03.pdf', metadata=True)

# ensure document texts are not bytes objects
for doc in documents:
    doc.text = doc.text.decode()

local_llm_path = './ggml-gpt4all-j-v1.3-groovy.bin'
llm = GPT4All(model=local_llm_path, backend='gptj', streaming=True, n_ctx=512)
llm_predictor = LLMPredictor(llm=llm)
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

prompt_helper = PromptHelper(max_input_size=512, num_output=256, max_chunk_overlap=-1000)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embed_model,
    prompt_helper=prompt_helper,
    node_parser=SimpleNodeParser(text_splitter=TokenTextSplitter(chunk_size=300, chunk_overlap=20))
)

index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist(persist_dir="./storage")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, service_context=service_context)
query_engine = index.as_query_engine(streaming=True, similarity_top_k=1, service_context=service_context)
response_stream = query_engine.query("What are the main climate risks to our Oceans?")
response_stream.print_response_stream()
