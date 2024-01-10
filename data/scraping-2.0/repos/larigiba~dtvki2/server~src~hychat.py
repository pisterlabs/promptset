import os
import torch
import pinecone
from operator import itemgetter

from src.templates import CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT
from src.utils import (
    FilteredRetriever,
    combine_documents,
    PineconeHybridSearchRetrieverWithScores,
)

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.memory import ConversationBufferMemory

from pinecone_text.sparse import SpladeEncoder

device = None
OPENAI_API_KEY = None


def init_env(openai_api_key, pinecone_api_key, pinecone_env):
    global OPENAI_API_KEY
    global device

    OPENAI_API_KEY = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # initialize pinecone
    pinecone.init(
        api_key=pinecone_api_key,  # "54ceda71-d9a7-44aa-b5e7-1cc49f403009",  # find at app.pinecone.io
        environment=pinecone_env,  # next to api key in console
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"


def get_chain():
    global memory
    global OPENAI_API_KEY
    # get dense model
    model_name = "text-embedding-ada-002"
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

    # get sparse model
    splade = SpladeEncoder()

    # get vectorstore inded
    index_name = "dakip"
    index = pinecone.Index(index_name)
    index.describe_index_stats()

    # get llm
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, model_name="gpt-4-1106-preview", temperature=0.0
    )

    # LangChain Retriever
    hybridretriever = PineconeHybridSearchRetrieverWithScores(
        embeddings=embed, sparse_encoder=splade, index=index
    )

    # Add metadata filtering via FilteredRetriever
    filtered_retriever = FilteredRetriever(vectorstore=hybridretriever)

    # First we add a step to load memory
    # This adds a "memory" key to the input object
    memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
    )

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables)
        | itemgetter("history"),
    )

    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    }

    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | filtered_retriever,
        "question": lambda x: x["standalone_question"],
    }

    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
        "docs": itemgetter("docs"),
    }

    # And now we put it all together!
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    return final_chain, memory
