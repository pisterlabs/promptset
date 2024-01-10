from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from api.llm.chains.conversation_retrieval.base import ConversationalRetrievalChain
from chromadb.utils import embedding_functions

from config import LLM_EMBEDDINGS

CHAT_MODEL = "gpt-4"
BASE_MODEL = "gpt-3.5-turbo"

# --------------- BASE LLMs -----------------
llm_chat = ChatOpenAI(
    model_name=CHAT_MODEL,
    verbose=False,
    request_timeout=240,
    temperature=0.5,
    streaming=True,
)
llm_condense = ChatOpenAI(
    model_name=BASE_MODEL,
    verbose=False,
    request_timeout=240,
    temperature=0.3,
    streaming=False,
)
llm_default = ChatOpenAI(
    model_name=BASE_MODEL,
    verbose=False,
    request_timeout=240,
    temperature=0.3,
    streaming=False,
)


class InstructorEmbedder(Embeddings):
    def __init__(self) -> None:
        super().__init__()
        self.embed_func = embedding_functions.InstructorEmbeddingFunction(
            model_name="hkunlp/instructor-large", device="cpu"
        )

    def embed_documents(
        self, texts: list[str], chunk_size: int | None = 0
    ) -> list[list[float]]:
        result = self.embed_func(texts)
        print("embedding documents", result)
        return result

    def embed_query(self, text: str) -> list[float]:
        result = self.embed_func([text])[0]
        return result


db = Chroma(
    collection_name="general-min_chunk_size",
    embedding_function=InstructorEmbedder(),
    persist_directory=str(LLM_EMBEDDINGS),
)
# db2 = chromadb.HttpClient(host="localhost", port=8000)
retriever = db.as_retriever(search_kwargs={"k": 10})
# print(db._client.get_collection("general-max-size-512").count())
# -------------- CHAINS ---------------
conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm_chat, retriever=retriever, condense_question_llm=llm_condense
)
