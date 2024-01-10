import os
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


business_data = """
opening_hours: "Monday to Friday 8:00 am to 5:00 pm, Saturday 8:00 am to 12:00 pm, break 12:00 pm to 1:00 pm",
phone_number: "1234567890",
address: "123 Main Street, City, State, Country",
email: "123@domain.com",
website: "www.domain.com",
payment_methods: "cash, credit card, debit card",
shipping_methods: "delivery, pickup",
return_policy: "30 days from purchase, must have receipt. Must be in original packaging."
"""

template = """You are a helpful assistant that responds to the user's question based on this content:
Content: {business_data}"""


system_prompt = PromptTemplate.from_template(
    template).format(business_data=business_data)

system_message_prompt = SystemMessagePromptTemplate.from_template(
    system_prompt)


human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])


# Set OpenAI LLM
llm_chat = ChatOpenAI(temperature=0.9, max_tokens=150,
                      model='gpt-3.5-turbo-0613', client='')

LLM_get_business_data = LLMChain(
    llm=llm_chat,
    prompt=chat_prompt
)


class GetBusinessDataInput(BaseModel):
    question: str = Field()


# LLM_get_business_data.predict(
#     content=business_data, text='Atiende los domingos y puedo pagar con tarjeta?')
