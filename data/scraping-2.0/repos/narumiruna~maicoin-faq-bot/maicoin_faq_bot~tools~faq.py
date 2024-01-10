from langchain.base_language import BaseLanguageModel
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain.vectorstores import FAISS
from loguru import logger

from ..document_loaders import FAQLoader


def load_faq_vectorstore(json_file: str = 'maicoin_faq_zh.json'):
    logger.info(f'loading json file: {json_file}')
    docs = FAQLoader().load_and_split(json_file)

    logger.info('creating vectorstore...')
    vectorstore = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())

    return vectorstore


def load_faq_tool(llm: BaseLanguageModel, json_file: str = 'maicoin_faq_zh.json', max_output_chars: int = 4000):
    vectorstore = load_faq_vectorstore(json_file=json_file)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True),
        verbose=True,
    )

    def _run(query):
        res = chain.run(query)[:max_output_chars]
        return res

    return Tool.from_function(
        name='MaiCoin-FAQ-chain',
        description=('Useful for when you need to answer questions. '
                     'You should ALWAYS use this. '
                     'Input should be a fully formed question.'),
        func=_run,
    )
