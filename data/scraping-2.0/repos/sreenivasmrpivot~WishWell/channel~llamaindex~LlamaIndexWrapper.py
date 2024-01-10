from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, ServiceContext, set_global_service_context, LLMPredictor
from llama_index.llms import HuggingFaceLLM, LlamaCPP
from llama_index.prompts.prompts import SimpleInputPrompt
# from sentence_transformers import SentenceTransformer
# from llama_index.embeddings import HuggingFaceEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from helper import get_document_path, get_embedding_model_path, get_model_path, get_tokenizer_path

from models import EmbeddingModelEnum, ModelLocationEnum, Wish

from common.logging_decorator import auto_log_entry_exit

@auto_log_entry_exit()
class LlamaIndexWrapper:

    # prepare the template we will use when prompting the AI
    system_prompt = """<<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>
    """

    query_wrapper_prompt = SimpleInputPrompt("[INST]{prompt}[/INST]")

    vectorStorePath = "./vector-store/llama-index"

    def __init__(self, wish: Wish):
        self.wish = wish
        self._load_embbedding()
        self._load_service_context()
        self._load_document()
        self._persist_document_inmemory()
        self._persist_document_ondisk()
        self._load_index_from_disk()
        self._load_query_engine()
        # self._load_chat_bot()

    def _load_embbedding(self):
        # self.embedding = HuggingFaceEmbedding(model_name=get_embedding_model_path(self.wish))
        # self.embedding = SentenceTransformer(get_embedding_model_path(self.wish))
        self.embedding = HuggingFaceEmbeddings(model_name=get_embedding_model_path(self.wish))
    
    def _load_service_context(self):
        if self.wish.modelLocation == ModelLocationEnum.local:
            llm = LlamaCPP(model_path=get_model_path(self.wish), verbose=True)
            llm_predictor = LLMPredictor(llm=llm)
            self.service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=self.embedding, chunk_size=1024, chunk_overlap=128)
        else:
            llm = HuggingFaceLLM(
                context_window=4096, 
                max_new_tokens=256,
                generate_kwargs={"temperature": 0.7, "do_sample": False},
                system_prompt=LlamaIndexWrapper.system_prompt,
                query_wrapper_prompt=LlamaIndexWrapper.query_wrapper_prompt,
                tokenizer_name=get_tokenizer_path(self.wish),
                model_name=get_model_path(self.wish),
                device_map="auto",
                stopping_ids=[50278, 50279, 50277, 1, 0],
                tokenizer_kwargs={"max_length": 4096},
                # uncomment this if using CUDA to reduce memory usage
                # model_kwargs={"torch_dtype": torch.float16}
            )
            self.service_context = ServiceContext.from_defaults(llm=llm, embed_model=self.embedding, chunk_size=1024, chunk_overlap=128)

    def _load_document(self):
        self.document = SimpleDirectoryReader(input_files=[get_document_path(self.wish)]).load_data()

    def _persist_document_inmemory(self):
        self.index = VectorStoreIndex.from_documents(self.document, service_context=self.service_context) # this is just in memory

    def _persist_document_ondisk(self):
        self.index.storage_context.persist(persist_dir=LlamaIndexWrapper.vectorStorePath) # this enables storing vector store on disk

    def _load_index_from_disk(self):
        self.storage_context = StorageContext.from_defaults(persist_dir=LlamaIndexWrapper.vectorStorePath)
        self.index = load_index_from_storage(storage_context=self.storage_context, service_context=self.service_context)

    def _load_query_engine(self):
        self.query_engine = self.index.as_query_engine()

    def _load_chat_bot(self):
        self.query_chat_bot = self.index.as_chat_bot()

    def run(self):
        output = self.query_engine.query(self.wish.whisper)
        return output.response
    
# if __name__ == '__main__':
#     wish = Wish(location="local", documentName="Business Conduct.pdf", modelName="Llama", channel="Llamaindex", whisper="what is Legal Holds?")
#     run(wish)