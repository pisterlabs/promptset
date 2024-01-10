import ray
import faiss
import pickle
import tiktoken
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from flows_new.config import get_logger


logger = get_logger(__name__)

class Component:
    def __init__(self, component_order=0, expects_input=False):
        self.output = None
        self.input_from_prev = None
        self.component_order = component_order
        self.expects_input = expects_input

    def _execute(self):
        raise NotImplementedError

    def execute(self):
        try:
            self._execute()
        except Exception as e:
            logger.error(f"Exception in component {self.component_order}: {e}")

    @ray.remote
    def pexecute(self):
        self.execute()
        return self.output


class PrintComponent(Component):
    def __init__(self, message=None, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def _execute(self):
        if self.expects_input:
            assert self.input_from_prev is not None, "Component expects input but none was provided"
            print(f"{self.component_order} message: {self.input_from_prev}")
            self.output = self.input_from_prev
        else:
            print(f"{self.component_order} message: {self.message}")
            self.output = self.message


class SquareComponent(Component):
    def __init__(self, number=None, **kwargs):
        super().__init__(**kwargs)
        self.number = number

    def _execute(self):
        if self.expects_input:
            assert self.input_from_prev is not None, "Component expects input but none was provided"
            self.output = self.input_from_prev ** 2
            print(f"{self.component_order} message: {self.output}")
        else:
            self.output = self.number ** 2
            print(f"{self.component_order} message: {self.output}")
            

class PDFReaderComponent(Component):
    def __init__(self, file_path, **kwargs):
        super().__init__(**kwargs)
        self.file_path = file_path
        self.loader = PyPDFLoader(file_path=self.file_path)

    def execute(self):
        self.output = self.loader.load()


class ChunkerComponent(Component):
    def __init__(self, chunk_size=1500, chunk_overlap=100, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def execute(self):
        assert self.input_from_prev is not None, "Component expects input but none was provided"
        self.output = self.splitter.split_documents(self.input_from_prev)


class OpenAIFAISSComponent(Component):
    def __init__(self, save_path, **kwargs):
        super().__init__(**kwargs)
        self.save_path = save_path

    def execute(self):
        # Assert self.input_from_prev is not None
        assert self.input_from_prev is not None, "Component expects input but none was provided"

        input = self.input_from_prev
        store = FAISS.from_documents(input, embedding=OpenAIEmbeddings())
        self.output = store

        if self.save_path is not None:
            faiss.write_index(store.index, f"{self.save_path}/docs.index")

            store.index = None
            
            # Pickle and store the VecDB.
            with open(f"{self.save_path}/faiss_store.pkl", "wb") as f:
                pickle.dump(store, f)


class LoadRetrieversComponent(Component):
    def __init__(self, path, k, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.k = 4

    def execute(self):
        index = faiss.read_index(f"{self.path}/docs.index")

        with open(f"{self.path}/faiss_store.pkl", "rb") as f:
            vectorstore = pickle.load(f)

        vectorstore.index = index

        self.output = vectorstore.as_retriever(search_kwargs={'k': self.k})


class RetrieverComponent(Component):
    def __init__(self, query, concatenate_docs, max_tokens=3000, enc_model="gpt-3.5-turbo", **kwargs):
        super().__init__(**kwargs)
        self.query = query
        self.concatenate_docs = concatenate_docs
        self.max_tokens = max_tokens
        
        self.enc_model = enc_model
        try:
            self.encoding = tiktoken.encoding_for_model(self.enc_model)
        except KeyError:
            logger.info(f"Encoding for model {self.enc_model} not found. Using default encoding.")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def execute(self):
        assert self.input_from_prev is not None, "Component expects input but none was provided"

        retriever = self.input_from_prev
        docs = retriever.get_relevant_documents(self.query)

        if self.concatenate_docs:
            self.output = self.concatenate_documents(docs)

    def concatenate_documents(self, documents):
        """Combine documents up to a certain token limit."""
        combined_docs = ""
        token_count = 0
        used_docs = []

        for doc in documents:
            doc_tokens = self.calculate_tokens(doc.page_content)
            if (token_count + doc_tokens) <= self.max_tokens:
                combined_docs += f"\n\n{doc.page_content}\nSource: {doc.metadata['source']}"
                token_count += doc_tokens
                used_docs.append(doc)

        return combined_docs, used_docs
    
    def calculate_tokens(self, document):
        """Calculate the number of tokens in a list of documents."""
        return len(self.encoding.encode(document))
    

class QandAComponent(Component):
    def __init__(self, query, system_message, user_message, **kwargs):
        super().__init__(**kwargs)
        self.query = query
        self.system_message = system_message
        self.user_message = user_message
        FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_message),
                HumanMessagePromptTemplate.from_template(self.user_message),
            ]
        )
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.)
        self.chain = LLMChain(llm=llm, prompt=FINAL_ANSWER_PROMPT)

    def execute(self):
        assert self.input_from_prev is not None, "Component expects input but none was provided"

        relevant_docs = self.input_from_prev
        self.output = self.chain.predict(question=self.query, document=relevant_docs)
