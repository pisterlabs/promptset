from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from llama_index.node_parser import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index import load_index_from_storage
from llama_index import (
    GPTVectorStoreIndex,
    LangchainEmbedding,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    download_loader,
    PromptHelper
)
import pyttsx3

class QuestionAnswering:
    

     def fetch_answers(self,question):
        PyMuPDFReader = download_loader("PyMuPDFReader")

        documents = PyMuPDFReader().load(file_path='star.pdf', metadata=True)


        local_llm_path = 'models/ggml-model-gpt4all-falcon-q4_0.bin'
        llm = GPT4All(model=local_llm_path, backend='gptj', streaming=True)
        llm_predictor = LLMPredictor(llm=llm)

        embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

        prompt_helper = PromptHelper(num_output=256)
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            prompt_helper=prompt_helper,
            node_parser=SimpleNodeParser(text_splitter=TokenTextSplitter(chunk_size=300, chunk_overlap=20))
        )
        # index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

        # index.storage_context.persist(persist_dir="./storage")

        """ Load the Index if already saved"""



        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context, service_context=service_context)


        query_engine = index.as_query_engine(streaming=True, similarity_top_k=1, service_context=service_context)
        

        response_stream = query_engine.query(f"{question}")
        # response_stream.print_response_stream()
      
        x=response_stream.get_response().response
        
        return x
     

    
