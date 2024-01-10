import openai
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from src.settings import OPENAI_API_KEY, configure_logger, template
from src.utils import CommaSeparatedListOutputParser

logger = configure_logger("Main components")

openai.api_key = OPENAI_API_KEY


system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_query = ""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_query)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)
chain = LLMChain(
    llm=ChatOpenAI(), prompt=chat_prompt, output_parser=CommaSeparatedListOutputParser()
)


def run_all(human_query: str, path: str):
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_query)
    logger.info("Created human prompt")
    loader = PyPDFDirectoryLoader(path)
    logger.info("Created loader")
    index = VectorstoreIndexCreator().from_loaders([loader])
    logger.info("Created index")
    logger.info(f"{human_message_prompt}")
    response = index.query(human_query)

    # qa_chain = RetrievalQA.from_chain_type(
    #    chain, retriever=index.vectorstore.as_retriever(), return_source_documents=True
    # )
    logger.info(response)
    return response
