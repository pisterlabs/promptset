import os
from dotenv import load_dotenv
from typing import Any, Dict, List

import logging

from langchain.vectorstores import Chroma
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, AgentType, Tool, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import AgentType

from messaging.telegram_integration import TelegramIntegration


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class TelegramBotHandler(BaseCallbackHandler):
    message = ""
    message_sent = ""

    def __init__(self, telegram_message_handler):
        self.telegram_message_handler = telegram_message_handler

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.message += token

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        self.telegram_message_handler.message.reply_text("...")

    def on_llm_end(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        message = ""
        message_sent = ""


class LLMBot:
    def __init__(self, telegram_bot, doc_chain):
        self.telegram_bot = telegram_bot
        self.doc_chain = doc_chain

    def setup_llm(self):
        # Create tools
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # LLM
        llm = ChatOpenAI(temperature=0)

        # Create the search tool (with DuckDuckGo)
        tools = load_tools(["google-serper", "llm-math"], llm=llm)
        tools += Tool(
            name="Paid leave QA System",
            func=self.doc_chain.run,
            description="Useful when you have a question about the paid leave in the company. Input should be a fully formed question.",
        ),
        self.search_agent = initialize_agent(
            tools, llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True, memory=memory, handle_errors=True)
        logger.info("LLM setup done")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = self.search_agent.run(
            update.message.text,
            #callbacks=[TelegramBotHandler(update)]
            )
        await update.message.reply_text(message)


def create_doc_qa_chain(doc: str):
    logger.info("Create doc qa chain")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(doc)

    from langchain.embeddings import OpenAIEmbeddings
    db = FAISS.from_documents(md_header_splits, OpenAIEmbeddings())
    llm = OpenAI(temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=db.as_retriever()
    )
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=db.as_retriever())
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=compression_retriever
    )


def main() -> None:

    # Firstly load all the environment variables
    load_dotenv()

    # Process doc
    with open("documents/conge_paye.md", "r") as f:
        doc = f.read()

    """Start the bot."""
    telegram_int = TelegramIntegration(bot_token=os.environ.get("TELEGRAM_TOKEN"))
    bot = LLMBot(telegram_int, create_doc_qa_chain(doc))
    bot.setup_llm()

    telegram_int.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    telegram_int.loop()


if __name__ == "__main__":
    main()
