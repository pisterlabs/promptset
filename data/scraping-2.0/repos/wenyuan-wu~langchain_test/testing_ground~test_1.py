from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.memory import ConversationBufferMemory


def chat_sum(text):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    template = "You are a helpful assistant that makes funny jokes."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    print(system_message_prompt)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # get a chat completion from the formatted messages
    messages = chat_prompt.format_prompt(text=text).to_messages()
    print(messages)
    result = chat(messages)
    return result


if __name__ == '__main__':
    load_dotenv()
    text = "Hi, what's the weather like?"
    result_sum = chat_sum(text)
    print(result_sum.content)
