"""Create a LangChain chain for question/answering."""
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Vectara
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser
from operator import itemgetter
import os


#
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> RetrievalQAWithSourcesChain:
    """Create a chain for question/answering."""

    load_dotenv()
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    hf_llm = HuggingFaceEndpoint(
        endpoint_url="https://euo6lqs9bqkddhci.us-east-1.aws.endpoints.huggingface.cloud",
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
        task="text-generation",
        model_kwargs={
            "temperature": 0.1,
            "max_new_tokens": 488,
        },
    )

    embeddings_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ["HF_TOKEN"],
        api_url="https://pikmjtam1n1c2rzu.us-east-1.aws.endpoints.huggingface.cloud",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt_template = """\
    Use the provided context to answer the user's question. If you don't know the answer, say you don't know.

    Context:
    {context}

    Question:
    {question}
    
    Answer in french and do not start with 'Réponse:'
    """

    rag_prompt = ChatPromptTemplate.from_template(prompt_template)

    entry_point_chain = {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    rag_chain = entry_point_chain | rag_prompt | hf_llm | StrOutputParser()

    rag_chain_with_sources = RunnableParallel(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
        "documents": lambda input: [doc.metadata for doc in input["documents"]],
        "answer": rag_chain,
    }
    return rag_chain_with_sources
