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


def chat_sum(text):
    chat = ChatOpenAI(temperature=0)
    template = "You are a helpful assistant that summarize the content to around 15% of it's original size. The " \
               "original content is from transcript of lectures, the summarization should be descriptive, instead of " \
               "like speech."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # get a chat completion from the formatted messages
    messages = chat_prompt.format_prompt(text=text).to_messages()
    # print(messages)
    result = chat(messages)
    return result


if __name__ == '__main__':
    load_dotenv()
    text = "You can make use of templating by using a MessagePromptTemplate. You can build a ChatPromptTemplate from " \
           "one or more MessagePromptTemplates. You can use ChatPromptTemplate’s format_prompt – this returns a " \
           "PromptValue, which you can convert to a string or Message object, depending on whether you want to use " \
           "the formatted value as input to an llm or chat model."
    result_sum = chat_sum(input("type input:\n"))
    print(result_sum.content)
