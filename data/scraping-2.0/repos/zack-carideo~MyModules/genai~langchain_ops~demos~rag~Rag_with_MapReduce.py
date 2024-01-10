# Import the necessary modules
import bs4
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.document_transformers import LongContextReorder, EmbeddingsClusteringFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_transformers import (
    EmbeddingsRedundantFilter,
)
from os.path import expanduser
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

#config 
db_dir = "/home/zjc1002/Mounts/data/" #directory to save vector indecies to disk
llm_dir = "/home/zjc1002/Mounts/llms/" #directory containing all local llms to use for retrieval , filtering, and generation 

#Foundational Model information (used for generation)
repo_id = "shaowenchen/llama-2-7b-langchain-chat-gguf"
filename = 'llama-2-7b-langchain-chat.Q4_K.gguf'
repo_type = "model"
local_dir = f"{llm_dir}/llama-2-7b-langchain-chat-gguf"
local_dir_use_symlinks = False
modelpath = Path(local_dir, filename) 
model_path = expanduser(modelpath)

#model memory settings
n_gpu_layers = 4 # Change this value based on your model and your GPU VRAM pool.
n_batch = 10    # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.


#source data information (this is what you are going to query / create RAG framework for)
source_data_dict={
    "source_data_url": "https://lilianweng.github.io/posts/2023-06-23-agent/"
    , "page_components_2_parse":["post-content", "post-title", "post-header"]
}

#define example queries to use to test framework (these are end user queries that are used for retrieval)
query_list = ["Is there any framework available to tackle the climate change?", "what is huggingGPT?",]

#summarization template for LLamaCpp
#template to use for generating summaries from long text 
template = """
Write a concise summary of the text, return your responses with 5 lines that cover the key points of the text.
```{text}```
SUMMARY:
"""

#Define the list of retrievers (include the foundatoinal model to use for filtering, in this case gpt2)
db_info = [{'db_name':'mini'
           , 'model_name':"sentence-transformers/all-MiniLM-L6-v2"
           , "model_kwargs": {"device":"cuda"}
           , "collection_meta":{"hnsw:space": "cosine"}
           , "persist_directory": f"{db_dir}/mini_tst"
           , "search_type": "similarity"
           , "search_kwargs": {"k":2, "include_metadata": True}
           , "filter":False

           }
           , {'db_name':'miniqa'
           , 'model_name':"sentence-transformers/multi-qa-MiniLM-L6-dot-v1"
           , "model_kwargs": {"device":"cuda"}
           , "collection_meta":{"hnsw:space": "cosine"}
           , "persist_directory": f"{db_dir}/miniqa_tst"
           , "search_type": "similarity"
           , "search_kwargs": {"k":2, "include_metadata": True}
           , "filter":False
           }
           , {'db_name':'bge'
           , 'model_name':"BAAI/bge-large-en-v1.5"
           , "model_kwargs": {"device":"cuda"}
           , "collection_meta":{"hnsw:space": "cosine"}
           , "persist_directory": f"{db_dir}/bge_tst"
           , "search_type": "similarity"
           , "search_kwargs": {"k":2, "include_metadata": True}
           , "filter":False
           }
           , {'db_name':'gpt2'
           , 'model_name':"gpt2"
           , "model_kwargs": {"device":"cuda"}
           , "filter":True
           }
           ]

#doc tokenization settings
doc_split_info = {'chunk_size':512
                  , 'chunk_overlap':100}

###
###START 
###

#LOAD SAMPLE DATA
#Define Sample Web Based Loader 
loader = WebBaseLoader(
    web_paths=(source_data_dict["source_data_url"],),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=(v for v in source_data_dict["page_components_2_parse"])
        )
    ),
)

#load and split sample docs 
docs = loader.load()

#define text chunking method
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=doc_split_info['chunk_size']
    , chunk_overlap=doc_split_info["chunk_overlap"])

#chunk text 
splits = text_splitter.split_documents(docs)
print(len(splits))


# load embeddings to use as retrievers
# use lots of small models to reduce bias and increase diversity of results
# NOTE: I wrap all models into dictionaries to enable flexible expansion/contraction of the number of retrievers used in the merged retriver 
embeddings_ = { _db["db_name"]:  HuggingFaceEmbeddings(model_name=_db['model_name']
                                    , model_kwargs = _db['model_kwargs']
                                    , cache_folder=llm_dir)
                                    for _db in db_info}

#Index documents using each of the retrievers defined above (using (Chroma))
vector_stores = {_db["db_name"]: Chroma.from_documents(splits
                                     , embeddings_[_db['db_name']]
                                     , collection_metadata=_db['collection_meta']
                                     , persist_directory=_db['persist_directory'])
                                     for _db in db_info if _db['filter'] == False}

#load teh vector indcies from disk for demo (use persist directory to prevent loading into memory)
#load the vector indecies
vec_indcies = {_db["db_name"]: Chroma(persist_directory= _db['persist_directory']
                                      , embedding_function=embeddings_[_db['db_name']])
                                      for _db in db_info   if _db['filter'] == False
}


#create retrievers from each vector index       
vecindex_retrievers = {_db["db_name"]: vec_indcies[_db['db_name']].as_retriever(
    search_type = _db["search_type"]
    , search_kwargs = _db['search_kwargs']
    )
    for _db in db_info if _db['filter'] == False
    }


#Generate LORD OF ALL RETRIEVERS (aka an ensemble of retrievers)
lotr = MergerRetriever(retrievers=[retrvr for retrvr in vecindex_retrievers.values()])


#the big model used for clusering and filtering results from merged retriver
filter_embeddings = [embeddings_[_db['model_name']] for _db in db_info if _db['filter'] == True][0]
filter_embeddings.client.tokenizer.pad_token = filter_embeddings.client.tokenizer.eos_token


# We can remove redundant results from both retrievers using yet another embedding.
# Using multiples embeddings in diff steps could help reduce biases.
# If you want the final document to be ordered by the original retriever scores
# you need to add the "sorted" parameter.
filter_ordered_by_retriever = EmbeddingsClusteringFilter(
    embeddings=filter_embeddings,
    num_clusters=3,
    num_closest=1,
    sorted=True,
)

# Compile final pipeline
# Contextual compression is a way of making it easy for models to fetch answers or relevant information 
# from the pool of data quickly. It allows the system to compress the files and filter out the irrelevant 
# information before making a similarity search or any kind of search. The compression is related to both 
# the data compression within the document and document compression from the pool of data:
# You can use an additional document transformer to reorder documents after removing redundance.
#filter = EmbeddingsRedundantFilter(embeddings=filter_embeddings)
reordering = LongContextReorder()
pipeline = DocumentCompressorPipeline(transformers=[ filter_ordered_by_retriever 
                                                    , reordering]
                                                    )

#final merged retriver to be used for IR 
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline
    , base_retriever=lotr
    , documents= splits
    )


#print top IR RESULTS (THIS IS NOT GENERATION)
for _query in query_list: 
    for chunks in compression_retriever.get_relevant_documents(_query):
        print(chunks.page_content)

##################
### GENERATION ###
##################

# USING LLAMACPP to illistrate how flexible langchain can be (you dont have to use OPENAI!)
llm = LlamaCpp(
    model_path=model_path
    , n_gpu_layers=n_gpu_layers
    , n_batch=n_batch
    , f16_kv=True  # MUST set to True, otherwise you will run into problem after a couple of calls
)

#use summarization prompt template to instruct llama2 to informally summarize the text
prompt = PromptTemplate(
    input_variables=['text'],
    template=template
)

#use summarization chain to generate summarry of all topn docs returned from merged retrueiver
chain = load_summarize_chain(
    llm,
    chain_type='stuff',
    prompt=prompt,
    verbose=False
)

#generate diffused awnsers for each original query in the query list 
results = []
for query_ in query_list: 
    #Retrieve topn documents from the merged retriever, these documents will be passed to LLamaccp for summarization / distillation into a single awnser 
    docs = compression_retriever.get_relevant_documents(query_list[0])

    #final augmented awnser 
    result = chain(docs)

    print(result['output_text'])

    results.append({'query':query_, 'awnser': result['output_text']})

print(results)
############
##END DEMO##
############