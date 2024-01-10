from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from invoice_chatbot.utils import set_qa_prompt
from langchain.chains import RetrievalQA



class InvoiceChatBot:
    def __init__(self, pdf_folder: str, emb_model_name_or_path: str, llm_model_name_or_path: str, model_type: str, max_new_tokens: int = 256, temperature: float = 0, device: str = "cpu") -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=emb_model_name_or_path,
                                            model_kwargs={'device': device})        
        self.llm = CTransformers(model=llm_model_name_or_path,
                            model_type=model_type,
                            config={'max_new_tokens': max_new_tokens,
                                    'temperature': temperature}
                            )
        self.pdf_folder = pdf_folder

    def __vectorize_data(self) -> None:
        loader = PyPDFDirectoryLoader(self.pdf_folder)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(docs)
        self.db = Chroma.from_documents(texts, self.embeddings, persist_directory="db")
    
    def chat_completion(self, query: str, return_source_document: bool = True, vector_count: int = 2) -> str:
        self.__vectorize_data()
        prompt = set_qa_prompt()
        dbqa = RetrievalQA.from_chain_type(llm=self.llm,
                                       chain_type='stuff',
                                       retriever=self.db.as_retriever(search_kwargs={'k': vector_count}),
                                       return_source_documents=return_source_document,
                                       chain_type_kwargs={'prompt': prompt}
                                       )

        return dbqa({'query': query})