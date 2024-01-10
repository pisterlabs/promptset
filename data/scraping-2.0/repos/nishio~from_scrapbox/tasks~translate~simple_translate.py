import openai
import dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0, model=llm_model)


SYSTEM_PROMPT = """\
You are translator. Translate single line Japanese input, which is title of a wiki page, to single line English."""

template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=(SYSTEM_PROMPT)),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)


def main(text):
    r = llm(template.format_messages(text=text))
    return r.content


if __name__ == "__main__":
    print(main("これは何？"))  # What is this?
