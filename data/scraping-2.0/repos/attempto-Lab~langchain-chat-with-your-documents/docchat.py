import os
import sys

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

# Alternative GPT4All
# from langchain.llms import GPT4All
# from langchain.embeddings import GPT4AllEmbeddings


from langchain.vectorstores import Chroma
# Alternative for small docs: use DocArrayInMemorySearch instead of Chroma for vector storage, see DocChat._load_db
# from langchain.vectorstores import DocArrayInMemorySearch

openai.api_key = os.environ['OPENAI_API_KEY']
llm_name = "gpt-3.5-turbo"

# Use OpenAI for chat completion
llm = ChatOpenAI(model_name=llm_name, temperature=0)
# Use OpenAI to create embeddings for chunks and questions
embeddings = OpenAIEmbeddings()

# Alternative GPT4All, model must point to the model-file you want to use
# llm = GPT4All(model="/home/alex/.local/share/nomic.ai/GPT4All/wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin", n_threads=8)
# embeddings = GPT4AllEmbeddings()

# create Vector Database to insert/query Embeddings - we currently do not persist the embeddings on disk
vectordb = Chroma(embedding_function=embeddings)


class Document:
    """
    Document class to store document name and filepath
    """
    def __init__(self, name: str, filepath: str):
        self.name = name
        self.filepath = filepath


class ChatResponse:
    """
    ChatResponse class to store the response from the chatbot.
    """
    answer: str = None
    db_question: str = None
    db_source_chunks: list = None

    def __init__(self, answer: str, db_question: str = None, db_source_chunks: list = None):
        self.answer = answer
        self.db_question = db_question
        self.db_source_chunks = db_source_chunks

    def __str__(self) -> str:
        return f"{self.db_question}: {self.answer} with {self.db_source_chunks}"


class DocChat:
    """
    DocChat class to store the info for the current document and provide methods for chat.
    """
    _qa: ConversationalRetrievalChain = None

    # string list of chat history
    _chat_history: list[(str, str)] = []

    # Holds current document, we can chat about
    doc: Document = None

    def process_doc(self, doc: Document) -> int:
        """
        Process the document (split, create & store embeddings) and create the chatbot qa-chain.
        :param doc:
        :return: number of embeddings created
        """
        self.doc = doc
        # For explanation of chain_type see https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a
        (self._qa, embeddings_count) = self._load_and_process_document(doc.filepath, chain_type="stuff", k=3)
        # Clear chat-history, so prompts and responses for previous documents are not considered as context
        self.clear_history()
        return embeddings_count

    @staticmethod
    def _load_and_process_document(file: str, chain_type: str, k: int) -> (ConversationalRetrievalChain, int):
        loader = PyPDFLoader(file)
        documents = loader.load()
        # Split document in chunks of 1000 chars with 150 chars overlap - adjust to your needs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        # Make sure that we forget the embeddings of previous documents
        db = Chroma()
        db.delete_collection()
        # Fill Vector Database with embeddings for current document
        db = Chroma.from_documents(docs, embeddings)

        # Alternative: see https://python.langchain.com/docs/integrations/vectorstores/docarray_in_memory
        # db = DocArrayInMemorySearch.from_documents(docs, embeddings)

        # Create Retriever, return `k` most similar chunks for each question
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        # create a chatbot chain, conversation-memory is managed externally
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
            verbose=True,
        )
        return qa, len(docs)

    def get_response(self, query) -> ChatResponse:
        if not query:
            return ChatResponse(answer="Please enter a question")
        if not self._qa:
            return ChatResponse(answer="Please upload a document first")
        # Let LLM generate the response and store it in the chat-history
        result = self._qa({"question": query, "chat_history": self._chat_history})
        answer = result["answer"]
        self._chat_history.extend([(query, answer)])
        return ChatResponse(answer, result["generated_question"], result["source_documents"])

    def clear_history(self):
        self._chat_history = []
        return


# main method for testing
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python docchat.py <file_path> <question>")
        sys.exit(1)

    file_path = sys.argv[1]
    question = sys.argv[2]

    doc_chat = DocChat()
    file_name = os.path.basename(file_path)
    embeddings_count = doc_chat.process_doc(Document(name=file_name, filepath=file_path))
    print(f"Created {embeddings_count} embeddings")
    response = doc_chat.get_response(question)
    print(response.answer)
    sources = [x.metadata for x in response.db_source_chunks]
    print(sorted(sources, key=lambda s: s['page']))
