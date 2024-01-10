import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


class LangChain:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    OPEN_AI_KEY = "YOUR KEY"
    os.environ['OPENAI_API_KEY'] = OPEN_AI_KEY

    def __init__(self, text=None, embeddings_type=None, chain_type="stuff"):
        self.chain = load_qa_chain(OpenAI(), chain_type=chain_type)
        if text is None or embeddings_type is None:
            self.textsearch = None
            self.data = None
        else:
            self.data = text
            self._build_textsearch(text, embeddings_type)

    def _build_textsearch(self, text, embeddings_type):
        if embeddings_type == "openai":
            embeddings = OpenAIEmbeddings()
        else:
            raise ValueError("Embeddings not supported")
        texts = self.text_splitter.split_text(text)
        self.textsearch = FAISS.from_texts(texts, embeddings)

    def chat(self, query):
        if self.textsearch is None:
            context = None
        else:
            context = self.textsearch.similarity_search(query)
        response = {"answer": self.chain.run(input_documents=context, question=query)}
        return response
