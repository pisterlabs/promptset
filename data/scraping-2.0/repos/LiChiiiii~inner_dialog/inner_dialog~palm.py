import json
from langchain import LLMChain
from langchain.chat_models import ChatGooglePalm
import google.generativeai
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from inner_dialog.utils import article2hiercc_palm, palm_summary
import openai
import os
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


def mult_round_palm(question: str, round: int = 3) -> str:
    """Directly give PALM the question. Nothing more.

    Args:
        question (str): The question to ask.

        round (int): How many round to ask PALM.

    Returns:
        str: PALM's response.
    """
    ans = ""
    chat = ChatGooglePalm(temperature=0.2)
    messages = [
        SystemMessage(content="You are a good summarizer."),
        HumanMessage(content=f"{question}"),
    ]
    for _ in range(round):
        msg = chat(messages)
        ans += palm_summary(msg.content) + "\n---------------\n"
        messages.extend(
            [
                msg,
                HumanMessage(content="Anything else?"),
            ]
        )
    return ans


def vanilla_palm(question: str) -> str:
    """Directly give PALM the question. Nothing more.

    Args:
        question (str): The question to ask.

    Returns:
        str: PALM's response.
    """
    chat = ChatGooglePalm(temperature=0.2)
    question_msg_temp = HumanMessagePromptTemplate.from_template(
        """
        Please give me at least 7 important things with "'{question}'" and give me some sub-things with each important things.
        """
    )
    messages = [
        SystemMessage(content="You are a helpful assistent."),
        question_msg_temp.format(question=question),
    ]
    palm_baseline = chat(messages)
    return palm_baseline.content


def t2cb_ask_palm(question: str) -> str:
    """Ask T2CB question to PALM.

    Args:
        question (str): Things to consider before ...

    Returns
        str: PALM's response.
    """
    response = vanilla_palm(question)
    # response = mult_round_palm(question, round=3)
    return response


if __name__ == "__main__":
    question = "things to consider before starting a business."
    article = t2cb_ask_palm(question)
    print(article)
    hierr = article2hiercc_palm(article)
    hierr_str = json.dumps(hierr, indent=2)
    print(hierr_str)
