from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from utils.constants import console as out
from utils.constants import ColorWrapper as CR
from AI.AIXBase import AIXBase
import os


class AIXEngine(AIXBase):
    def __init__(self) -> None:
        super().__init__()

    def init_engine(self) -> bool:
        # super().__init__()
        out(msg="OAIX Init", color=CR.green, reset=True)

        self.embedding = OpenAIEmbeddings()
        out(msg="OAIX Reason engine", color=CR.green, reset=True)
        if not self.embedding:
            out(msg=f"embedding object not initiated", color=CR.red, reset=True)
            return False

        out(msg=f"OAIX executing cwd {os.getcwd()}", color=CR.blue, reset=True)

        self.loader = DirectoryLoader("news", glob="**/*.txt")
        out(msg="OAIX reading documents", color=CR.green, reset=True)
        if not self.loader:
            out(msg=f"loader object not initiated", color=CR.red, reset=True)
            return False

        self.documents = self.loader.load()
        out(
            msg=f"OAIX documents read {len(self.documents)}", color=CR.green, reset=True
        )
        if not self.documents:
            out(msg=f"document object not initiated", color=CR.red, reset=True)
            return False

        self.text_splitter = CharacterTextSplitter(
            chunk_size=2500, chunk_overlap=0)
        out(msg=f"OAIX splitting documents into chunks", color=CR.green, reset=True)
        if not self.text_splitter:
            out(msg=f"text_splitter object not initiated", color=CR.red, reset=True)
            return False

        self.texts = self.text_splitter.split_documents(self.documents)
        if not self.texts:
            out(msg=f"texts object not initiated", color=CR.red, reset=True)
            return False

        self.vec_store = Chroma.from_documents(self.texts, self.embedding)
        out(msg=f"OAIX create vectors", color=CR.green, reset=True)
        if not self.vec_store:
            out(msg=f"vec_store object not initiated", color=CR.red, reset=True)
            return False

        self.qa = RetrievalQA.from_chain_type(
            llm=OpenAI(), chain_type="stuff", retriever=self.vec_store.as_retriever()
        )
        out(msg=f"OAIX ready", color=CR.green, reset=True)
        if not self.qa:
            out(msg=f"qa object not initiated", color=CR.red, reset=True)
            return False

        return True

    def prompt(self, **kwargs) -> any:
        if not self.vec_store:
            out(msg=f"vec_store object not initiated", color=CR.red, reset=True)
            return "qa failed ot init"

        input = kwargs["input"]
        return self.qa.run(input)
