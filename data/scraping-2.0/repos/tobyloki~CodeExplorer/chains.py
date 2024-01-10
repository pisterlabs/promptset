from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import (
    OllamaEmbeddings,
    # SentenceTransformerEmbeddings,
    BedrockEmbeddings,
)
from langchain.chat_models import ChatOpenAI, ChatOllama, BedrockChat
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from typing import List, Any
from utils import BaseLogger
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


def load_embedding_model(embedding_model_name: str, logger=BaseLogger(), config={}):
    if embedding_model_name == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=config["ollama_base_url"], model="codellama:7b-instruct"
        )
        dimension = 4096
        logger.info("Embedding: Using Ollama")
    elif embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using OpenAI")
    elif embedding_model_name == "aws":
        embeddings = BedrockEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using AWS")
    # else:
    #     embeddings = SentenceTransformerEmbeddings(
    #         model_name="all-MiniLM-L6-v2", cache_folder="./embedding_model"
    #     )
    #     dimension = 384
    #     logger.info("Embedding: Using SentenceTransformer")
    return embeddings, dimension


def load_llm(llm_name: str, logger=BaseLogger(), config={}):
    if llm_name == "gpt-4":
        logger.info("LLM: Using GPT-4")
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    elif llm_name == "gpt-3.5":
        logger.info("LLM: Using GPT-3.5")
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    elif llm_name == "claudev2":
        logger.info("LLM: ClaudeV2")
        return BedrockChat(
            model_id="anthropic.claude-v2",
            model_kwargs={"temperature": 0.0, "max_tokens_to_sample": 1024},
            streaming=True,
        )
    elif len(llm_name):
        logger.info(f"LLM: Using Ollama: {llm_name}")
        return ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=llm_name,
            streaming=True,
            # seed=2,
            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=3072,  # Sets the size of the context window used to generate the next token.
        )
    logger.info("LLM: Using GPT-3.5")
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)


def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that helps a support agent with answering programming questions.
    If you don't know the answer, just say that you don't know, you must not make up an answer.
    """
    human_template = "{question}"

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),        # The persistent system prompt
        MessagesPlaceholder(variable_name="chat_history"),          # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(human_template)    # Where the human input will injected
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        verbose=False,
        memory=memory,
    )

    def generate_llm_output(
        user_input: str, callbacks: List[Any]
    ) -> str:    
        answer = chain.invoke(user_input, config={"callbacks": callbacks})["text"]
        return answer

    return generate_llm_output


def get_qa_rag_chain(_vectorstore, llm):
    # Create qa RAG chain
    system_template = """ 
    Use the following pieces of context to answer the question at the end.
    The context contains code source files which can be used to answer the question as well as be used as references.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    {summaries}
    ----
    Generate concise answers with references to code source files at the end of every answer.
    """
    user_template = "Question:```{question}```"
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template), # The persistent system prompt
        HumanMessagePromptTemplate.from_template(user_template),    # Where the human input will injected
    ])
    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=chat_prompt,
    )
    qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 2}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=3375,
        return_source_documents=True
    )

    return qa