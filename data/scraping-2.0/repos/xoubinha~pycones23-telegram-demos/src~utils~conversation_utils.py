import os
from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


DEFAULT_SYSTEM_PROMPT = "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."


def generate_conversation(model_name: str = "gpt-3.5-turbo", openai_api_key=None, system_prompt: str = None) -> LLMChain:
    if openai_api_key is None:
        openai_api_key = os.environ["OPENAI_API_KEY"]
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    llm = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key)

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                system_prompt
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    return conversation


def conversate(id: str, conversations: dict, message: str, system_prompt: str = None) -> str:
    if id in conversations:
        conversation = conversations[id]
    else:
        conversation = generate_conversation(system_prompt=system_prompt)
        conversations[id] = conversation
    response = conversation(message)
    reply = response["text"]
    return reply


def end_conversation(id: str, conversations: dict):
    if id in conversations:
        del conversations[id]
        return True
    else:
        return False
