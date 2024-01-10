import uuid
from tempfile import NamedTemporaryFile
from typing import Tuple, List, Optional, Dict

from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chat_models import (
    AzureChatOpenAI,
    ChatOpenAI,
    ChatAnthropic,
    ChatAnyscale,
)
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import Document, BaseRetriever
from langchain.schema.chat_history import BaseChatMessageHistory
from langchain.schema.runnable import RunnablePassthrough
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.base import BaseTool
from langchain.vectorstores import FAISS
from langchain_core.messages import SystemMessage

from defaults import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_RETRIEVER_K
from qagen import get_rag_qa_gen_chain
from summarize import get_rag_summarization_chain


def get_agent(
    tools: list[BaseTool],
    chat_history: BaseChatMessageHistory,
    llm: BaseLLM,
    callbacks,
):
    memory_key = "agent_history"
    system_message = SystemMessage(
        content=(
            "Do your best to answer the questions. "
            "Feel free to use any tools available to look up "
            "relevant information, only if necessary"
        ),
    )
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
    )
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    # agent_memory = AgentTokenBufferMemory(
    #     chat_memory=chat_history,
    #     memory_key=memory_key,
    #     llm=llm,
    # )

    agent_memory = ConversationBufferMemory(
        chat_memory=chat_history,
        return_messages=True,
        memory_key=memory_key,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=agent_memory,
        verbose=True,
        return_intermediate_steps=False,
        callbacks=callbacks,
    )
    return (
        {"input": RunnablePassthrough()}
        | agent_executor
        | (lambda output: output["output"])
    )


def get_doc_agent(
    tools: list[BaseTool],
    llm: Optional[BaseLLM] = None,
    agent_type: AgentType = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
):
    if llm is None:
        llm = ChatOpenAI(
            model_name="gpt-4-1106-preview",
            temperature=0.0,
            streaming=True,
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You assist a chatbot with answering questions about a document.
                If necessary, break up incoming questions into multiple parts,
                and use the tools provided to answer smaller questions before
                answering the larger question.
                """,
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ],
    )
    agent_executor = initialize_agent(
        tools,
        llm,
        agent=agent_type,
        verbose=True,
        memory=None,
        handle_parsing_errors=True,
        prompt=prompt,
    )
    return (
        {"input": RunnablePassthrough()}
        | agent_executor
        | (lambda output: output["output"])
    )


def get_runnable(
    use_document_chat: bool,
    document_chat_chain_type: str,
    llm,
    retriever,
    memory,
    chat_prompt,
    summarization_prompt,
):
    if not use_document_chat:
        return LLMChain(
            prompt=chat_prompt,
            llm=llm,
            memory=memory,
        ) | (lambda output: output["text"])

    if document_chat_chain_type == "Q&A Generation":
        return get_rag_qa_gen_chain(
            retriever,
            llm,
        )
    elif document_chat_chain_type == "Summarization":
        return get_rag_summarization_chain(
            summarization_prompt,
            retriever,
            llm,
        )
    else:
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=document_chat_chain_type,
            retriever=retriever,
            output_key="output_text",
        ) | (lambda output: output["output_text"])


def get_llm(
    provider: str,
    model: str,
    provider_api_key: str,
    temperature: float,
    max_tokens: int,
    azure_available: bool,
    azure_dict: dict[str, str],
):
    if azure_available and provider == "Azure OpenAI":
        return AzureChatOpenAI(
            azure_endpoint=azure_dict["AZURE_OPENAI_BASE_URL"],
            openai_api_version=azure_dict["AZURE_OPENAI_API_VERSION"],
            deployment_name=azure_dict["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_key=azure_dict["AZURE_OPENAI_API_KEY"],
            openai_api_type="azure",
            model_version=azure_dict["AZURE_OPENAI_MODEL_VERSION"],
            temperature=temperature,
            streaming=True,
            max_tokens=max_tokens,
        )

    elif provider_api_key:
        if provider == "OpenAI":
            return ChatOpenAI(
                model_name=model,
                openai_api_key=provider_api_key,
                temperature=temperature,
                streaming=True,
                max_tokens=max_tokens,
            )

        elif provider == "Anthropic":
            return ChatAnthropic(
                model=model,
                anthropic_api_key=provider_api_key,
                temperature=temperature,
                streaming=True,
                max_tokens_to_sample=max_tokens,
            )

        elif provider == "Anyscale Endpoints":
            return ChatAnyscale(
                model_name=model,
                anyscale_api_key=provider_api_key,
                temperature=temperature,
                streaming=True,
                max_tokens=max_tokens,
            )

    return None


def get_texts_and_multiretriever(
    uploaded_file_bytes: bytes,
    openai_api_key: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    k: int = DEFAULT_RETRIEVER_K,
    azure_kwargs: Optional[Dict[str, str]] = None,
    use_azure: bool = False,
) -> Tuple[List[Document], BaseRetriever]:
    with NamedTemporaryFile() as temp_file:
        temp_file.write(uploaded_file_bytes)
        temp_file.seek(0)

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=0,
        )
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

        texts = text_splitter.split_documents(documents)
        id_key = "doc_id"

        text_ids = [str(uuid.uuid4()) for _ in texts]
        sub_texts = []
        for i, text in enumerate(texts):
            _id = text_ids[i]
            _sub_texts = child_text_splitter.split_documents([text])
            for _text in _sub_texts:
                _text.metadata[id_key] = _id
            sub_texts.extend(_sub_texts)

        embeddings_kwargs = {"openai_api_key": openai_api_key}
        if use_azure and azure_kwargs:
            azure_kwargs["azure_endpoint"] = azure_kwargs.pop("openai_api_base")
            embeddings_kwargs.update(azure_kwargs)
            embeddings = AzureOpenAIEmbeddings(**embeddings_kwargs)
        else:
            embeddings = OpenAIEmbeddings(**embeddings_kwargs)
        store = InMemoryStore()

        # MultiVectorRetriever
        multivectorstore = FAISS.from_documents(sub_texts, embeddings)
        multivector_retriever = MultiVectorRetriever(
            vectorstore=multivectorstore,
            docstore=store,
            id_key=id_key,
        )
        multivector_retriever.docstore.mset(list(zip(text_ids, texts)))
        # multivector_retriever.k = k

        multiquery_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # MultiQueryRetriever
        multiquery_texts = multiquery_text_splitter.split_documents(documents)
        multiquerystore = FAISS.from_documents(multiquery_texts, embeddings)
        multiquery_retriever = MultiQueryRetriever.from_llm(
            retriever=multiquerystore.as_retriever(search_kwargs={"k": k}),
            llm=ChatOpenAI(),
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[multiquery_retriever, multivector_retriever],
            weights=[0.5, 0.5],
        )
        return multiquery_texts, ensemble_retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
