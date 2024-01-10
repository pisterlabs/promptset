from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate


def llm_parse(input_text, temperature=0.9, max_tokens=3000):
    chat = ChatOpenAI(temperature=temperature)

    system_message_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template="For the following json objet, please help me parse the following infotmation and \
                save into a json object with the following keys \
                    (job_title, company_name, location, source, \
                        full_text_description, job_type(full_time/ part_time/ contract/ ...), link)",
            input_variables=[],
        )
    )

    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="The following is the parsed job description json object {json}?",
            input_variables=["json"],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)
    parsed_jd = chain.run(json=input_text)

    return parsed_jd
