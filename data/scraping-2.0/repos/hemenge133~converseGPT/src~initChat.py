from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate

def initChat(api_key):
    systemprompt="You are a helpful assistant."

    model_kwargs = {"top_p": 0.8, "frequency_penalty": 0.2, "presence_penalty": 0.1}

    chat = ChatOpenAI(model_name="gpt-4", temperature=0.4, model_kwargs=model_kwargs, openai_api_key=api_key)

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(systemprompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{message}")
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(
        llm=chat,
        prompt=prompt,
        memory=memory
    )
    return memory, conversation
