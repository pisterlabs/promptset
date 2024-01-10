from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from tqdm import tqdm
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0)

assignments = [
    "adelsberger.txt",
    "allison.txt",
    "bunde.txt",
    "heidt.txt",
    "kampwirth.txt"
    ]


def make_question_template(ask=True, extra=False):
    template = (
        "You are a good student who does all their homework to the best of their ability. Produce an answer to the following assignment prompt that might get an A. Format your answers in markdown."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{assignment}"
    if ask:
        human_template+="\n{ask}"
    if extra:
        human_template+="\n{extra}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    return = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

def make_ask_template():
    template = (
        "You are a good student who does all their homework to the best of their ability. Succinctly summarize what the assignment is asking for. Format your answers in markdown."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{assignment}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    return ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

def with_ask():
    for a in tqdm(assignments):
        with open(f"assignments/{a}") as asn:
            assignment = asn.read()
        ask = chat(make_ask_template().format_prompt(assignment=assignment).to_messages())
        result = chat(make_question_template().format_prompt(assignment=assignment, ask=ask).to_messages())
        with open(f"ask/{a[:4]}.md", 'w') as out:
            out.write("# Ask\n" + ask.content + "\n\n# Answer\n\n" + result.content)

def just_answer():
    for a in tqdm(assignments):
        with open(f"assignments/{a}") as asn:
            assignment = asn.read()
        result = chat(make_question_template(ask=False).format_prompt(assignment=assignment).to_messages())
        with open(f"answer/{a[:4]}.md", 'w') as out:
            out.write("# Answer\n\n" + result.content)

def no_prompt():
    human_template = "{assignment}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    no_prompt = ChatPromptTemplate.from_messages(
        [human_message_prompt]
    )
    for a in tqdm(assignments):
        with open(f"assignments/{a}") as asn:
            assignment = asn.read()
        result = chat(no_prompt.format_prompt(assignment=assignment).to_messages())
        with open(f"example/{a[:-4]}.md", 'w') as out:
            out.write(result.content)

if __name__ == "__main__":
    no_prompt()