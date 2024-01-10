from langchain.vectorstores import Chroma 
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from langchain.chains import LLMChain, RetrievalQA
from langchain.schema import Document
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model
from typing import List, Iterable
from pydantic import BaseModel, Field
import json, logging, configparser, os


#logging.basicConfig()
#logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
#logging.getLogger("langchain.retrievers.document_compressors.embeddings_filter").setLevel(logging.INFO)

#def web_scrap(url):
#    loader = RecursiveUrlLoader(url=url, extractor=lambda x: Soup(x, "html.parser").text)
#    docs = loader.load
#    return docs

#weird dependecy on dill
#def load_data_from_huggingface(path,name=None,page_content_column='text', max_len=20):
#    #LangChain Wrapper does not support splits and assumes text context https://github.com/langchain-ai/langchain/issues/10674
#    from langchain.document_loaders import HuggingFaceDatasetLoader
#    loader = HuggingFaceDatasetLoader(path, page_content_column, name)
#    docs = loader.load()
#    if len(docs) < max_len:
#        return docs[:max_len]
#    else:
#        return docs

def load_vector_db(path):
    from langchain.embeddings import HuggingFaceBgeEmbeddings #requires sentence-transformers
    # embed and load it into Chroma
    encoder_model= "BAAI/bge-small-en-v1.5"
    device ='cpu'
    normalize_embeddings = True
    model_name = encoder_model
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': normalize_embeddings}
    
    bge_hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    embedding_function= bge_hf
    db = Chroma(persist_directory=path, embedding_function = embedding_function)
    db.persist()
    return db

def create_vector_db(docs):
    #Assumes input in LangChain Doc Format
    #[Document(page_content='...', metadata={'source': '...'})]
    #split docs into chunks
    from langchain.text_splitter import TokenTextSplitter #requires tiktoken
    from transformers import AutoTokenizer
    encoder_model= "BAAI/bge-base-en"
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer,chunk_size=3000, chunk_overlap=0,separators = ["\n\n", "\n"], keep_separator= False)
    docs_split = text_splitter.transform_documents(docs)
    
    ###Get Encodder Model
    from langchain.embeddings import HuggingFaceBgeEmbeddings #requires sentence-transformers
    
    #This was not working in Wx Notebook
    #try:
    #    import torch
    #    is_cuda_available = torch.cuda.is_available()
    #except ImportError:
    #    is_cuda_available = False

    #device = 'cuda' if is_cuda_available else 'cpu'
    #normalize_embeddings = is_cuda_available
    device ='cpu'
    normalize_embeddings = True
    
    
    model_name = encoder_model
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': normalize_embeddings}
    
    bge_hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    embedding_function= bge_hf
    
    # embed and load it into Chroma
    db = Chroma.from_documents(docs_split, embedding_function,persist_directory="./chroma_db")
    db.persist()
    return db

# Output parser will split the LLM result into a list of queries
#Pydantic requires classes
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


class Credentials:
    def __init__(self,cfg_file=None, key=None, project=None, url=None):
        if cfg_file:
                self._load_credentials_from_file(cfg_file)
        # If key and project and url are provided, set attributes directly.
        elif key and project and url:
            self.watsonx_project_id = project
            self.watsonx_key = key
            self.watsonx_url = url
            
    def _load_credentials_from_file(self, cfg_file):
        config = configparser.ConfigParser()
        
        # Check if file exists.
        file_path = os.path.join(os.path.dirname(__file__), cfg_file)
        if not os.path.exists(file_path):
            raise SystemExit(f"Configuration file '{cfg_file}' does not exist.")
        
        config.read(file_path)
        
        try:
            self.watsonx_project_id = config['WATSONX_CREDS']['project_id']
            self.watsonx_key = config['WATSONX_CREDS']['key']
            self.watsonx_url = config['WATSONX_CREDS']['url']
        except KeyError:
            raise SystemExit("Could not find key in the configuration file.")



class RAGUtils:
    
    def __init__(self, db, credentials: Credentials = None):
        self._db = db
        self._creds = credentials
        #self._history = ChatMessageHistory()
        #Load Chain
        try:
            #Low Temp Model Low Output (Faster/Cheaper) Model for RAG 
            self._llm_llama_temp_low = self._get_llama(temp=0,max_new_tokens=200,top_k=1,top_p=0, repetition_penality = 1)
            #For Test self._llm_llama_temp_low = self._get_llama(temp=0.2,max_new_tokens=200,top_k=10,top_p=0.2)
            #get a mid temp high output model for the QA
            self._llm_llama_temp_high = self._get_llama(temp=0.2,max_new_tokens=150,top_k=25,top_p=.5, repetition_penality = 1)
            #The RAG
            self._chain = self._load_chain()
        except Exception as e:
            raise SystemExit(f"Error loading chain: {e}")

    #For use with front end can call chain below for testing without app
    def __call__(self,inp,history):#, chat_history):
        output_dict = self._chain(inp)
        #print(output_dict)
        output_str = f"""Answer: {output_dict['result']}"""
        for doc in output_dict['source_documents']:
          page_content_str = f"-------Page Content------\n\n{doc.page_content}"
          source_str = f"--------Source-----------\n\n{doc.metadata['source']}"
          output_str = f"{output_str}\n\n{source_str}\n\n{page_content_str}"
        output = output_str
        # Update the history
        history.append((inp, output))
        return "", history
        # print(output)
 
    def _get_llama(self,temp, max_new_tokens, top_k, top_p, repetition_penality):
        # Base on LLAMA tuning which seemed to greater variable than GPT or other WatsonX Foundation Models:
        # Found  that SAMPLE was better than GREEDY to stay on task and get stop tokens BUT Needed a lot of fine tuning
        # Low Temp LLM in init should be GREEDY like but SAMPLE with Low TEMP Parameters worked better
        # Tuning Temp would create BIG change but higher than .5 it was hallucinate (become TOO creative) which for a different application might be goood
        # Anthing hight than temp = 0 in long run testing it was noticed the intructions would sometimes be ingored
        # Tuning top_k, top_p  parameters could fine tune the big jumps from temp
        # Also Controling Max New Tokens was better at getting it to stop then stop tokens or repetition penaltiy
        params = {
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
            GenParams.TEMPERATURE: temp,
            GenParams.TOP_K: top_k,
            GenParams.TOP_P: top_p,
            #GenParams.RANDOM_SEED: 1234,
            GenParams.REPETITION_PENALTY: repetition_penality
        }

      
        llama = ModelTypes.LLAMA_2_70B_CHAT #oh yeah

        llama_model= Model(
            model_id=llama,
            params=params,
            credentials={'url':self._creds.watsonx_url,'apikey':self._creds.watsonx_key},
            project_id=self._creds.watsonx_project_id)
          
        #LangChain Ready
        return WatsonxLLM(model=llama_model)
    
    def _get_retriever_chain(self):      
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            #temp = 0 it would all most never return the orignal question.
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given question. Provide these alternative questions seperated by newlines.
            Include the original question in the returned list. Only use the question provided. Do not add any further intructions or information.
            Original question:{question}""",
        )

        base_retriever = self._db.as_retriever()
        embeddings = self._db.embeddings
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
        compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=base_retriever)

        #too slow
        #compressor = LLMChainExtractor.from_llm(llm_llama_temp_0p1)
        #compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

        output_parser = LineListOutputParser()
        
        retriever_llm_chain = LLMChain(llm=self._llm_llama_temp_low, prompt=QUERY_PROMPT, output_parser=output_parser)
        multi_retriever = MultiQueryRetriever(
            retriever=compression_retriever, llm_chain=retriever_llm_chain, parser_key="lines"
        )  # "lines" is the key (attribute name) of the parsed output
        return multi_retriever
    

    def _load_chain(self):
        qa_template = (
            """Use the following pieces of text delimited by triple
        backticks to answer the question delimited by double backticks.
        Use as much information as you can to create a dense summary. 
        Limit your answer to 3-5 sentences.
        Do not make up questions. If no context is provided, 
        just say No context found.
        ```{context}```

        ``{question}``

        Helpful Answer:

        """
        )
        qa_kwargs={
            "prompt": PromptTemplate(
            template=qa_template,
            input_variables=["context", "question"])
        }
        
        chain = RetrievalQA.from_chain_type(
            chain_type = "stuff",
            chain_type_kwargs = qa_kwargs,
            llm = self._llm_llama_temp_high,
            retriever = self._get_retriever_chain(),
            return_source_documents = True,
        )
        return chain


#For App Testing
def _get_model_reply(message, chat_history):
   bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
   chat_history.append((message, bot_message))
   time.sleep(1)
   print(chat_history)
   return "", chat_history

