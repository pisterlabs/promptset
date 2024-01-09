from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains import LLMChain


def init_llm_chain(prompt_template, template_variables, chat_temperature, output_key):
    human_messsage_prompt_template = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=prompt_template,
            input_variables=[var for var in template_variables],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [human_messsage_prompt_template])
    chat = ChatOpenAI(temperature=chat_temperature)
    return LLMChain(llm=chat, prompt=chat_prompt_template, output_key=output_key, verbose=True)

