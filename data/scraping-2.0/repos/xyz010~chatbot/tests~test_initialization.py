# Test the initialization of the ChatOpenAI object: Verify that the max_tokens and temperature parameters are set correctly.
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)


def test_chat_openai_init():
    chat_openai = ChatOpenAI(max_tokens=150, temperature=0.9)
    assert chat_openai.max_tokens == 150
    assert chat_openai.temperature == 0.9

# Test the initialization of the ChatPromptTemplate object: Verify that the system message, messages placeholder, and human message template are set correctly.

def test_prompt_is_ChatPromptTemplate():
    field_of_work = "plumbing"
    experience = 3
    system_message = "You are a door-to-door salesperson expert in {field_of_work}. You are teaching someone with {experience} years of experience how to be an effective door-to-door salesperson."
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    assert isinstance(prompt, ChatPromptTemplate)