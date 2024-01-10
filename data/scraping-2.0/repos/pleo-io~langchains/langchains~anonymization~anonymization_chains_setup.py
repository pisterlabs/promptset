from langchains.anonymization.anonymization_chain import AnonymizationChain
from langchains.anonymization.de_anonymization_chain import DeAnonymizationChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import SimpleSequentialChain, ConversationChain
from logging import Logger


def setup_anonymization_chains(key: bytes):
    anonymization_chain = AnonymizationChain(key=key, verbose=True)
    de_anonymization_chain = DeAnonymizationChain(key=key, verbose=True)
    return anonymization_chain, de_anonymization_chain


def setup_conversational_chain(
    anonymization_chain: AnonymizationChain,
    de_anonymization_chain: DeAnonymizationChain,
    prompt: PromptTemplate,
    llm: BaseLLM,
    memory: ConversationBufferMemory,
    logger: Logger,
):
    logger.info("Setting up LangChain...")

    chain = SimpleSequentialChain(
        chains=[
            anonymization_chain,
            ConversationChain(
                llm=llm,
                prompt=prompt,
                verbose=True,
                memory=memory,
                input_key="anonymized",
            ),
            de_anonymization_chain,
        ]
    )

    logger.info("Set up LangChain")
    return chain
