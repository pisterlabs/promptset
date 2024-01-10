from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
import datetime


class QAParam(BaseModel):
    question: str
    context: str
    datetime: str = str(datetime.date.today())
    # output_format: str


QA_PROMPT = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.\n
The output should be formatted as a JSON instance that conforms to the JSON schema below.
summary: str
As an example, for the schema
{{"answer": ""}}\n
Question: {question} \nContext: {context} \nAnswer:

"""

QAPromptTemplate = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are ChatGPT, a large language model trained by OpenAI.\n"
            "Knowledge cutoff: 2022-01\n"
            "Current date: {datetime}"
        ),
        HumanMessagePromptTemplate.from_template(QA_PROMPT),
    ]
)
