from langchain.chains import ConversationChain, LLMChain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pymongo.collection import Collection
from langchain.memory import ConversationBufferMemory


def mock_langchain(name: str):
    pass


def matching():
    # Data we get: Name
    # TODO: given Nikhils function, generate LangChain agents

    # TODO: iterate over all agents
    # Nikhil gives me the system prompt
    llm_one = OpenAI(
        temperature=0,
        openai_api_key="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",
    )
    llm_two = OpenAI(
        temperature=0,
        openai_api_key="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",
    )
    prompt = SystemMessage(
        content="You are a personal assistent. End each message with: assistant finished."
    )
    DEFAULT_TEMPLATE = """ This is a conversation between two pirates

        Current conversation:
        {history}
        Human: {input}
        AI:"""
    conversation_one, conversation_two = converse(
        llm_one, llm_two, template=DEFAULT_TEMPLATE, conversation_length=15
    )  # Initial promt might be


def converse(llm_one, llm_two, template, conversation_length):
    """
    Given two LLMs, this returns the conversation.

    :param llm_one:
    :param llm_two:
    :return:
    """
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

    conversation_one = ConversationChain(
        llm=llm_one,
        verbose=True,
        memory=ConversationBufferMemory(),
        prompt=PROMPT,
    )
    conversation_two = ConversationChain(
        llm=llm_two,
        verbose=True,
        memory=ConversationBufferMemory(),
        prompt=PROMPT,
    )
    output_two = "Hey there, let's get to know each other!"
    # System: those are my intersts -> system message for both of them
    for i in range(conversation_length):
        output_one = conversation_one.predict(input=output_two)
        output_two = conversation_two.predict(input=output_one)

    return conversation_one, conversation_two


matching()
