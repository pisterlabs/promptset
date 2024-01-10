from datetime import datetime
from typing import Any

from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .bard_llm import BardLLM


def summarize_text(corpora: str) -> str:
    """Summarize the text provided using langchain

    Arguments:
        corpora {str} -- The text to summarize

    Returns:
        str -- The summary of the text
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
    texts = text_splitter.split_text(corpora)
    docs = [Document(page_content=t) for t in texts]

    map_str = """Ignore all previous instructions. You are given a text about news happening in the space of artifical intelligence, machine learning or data science.
    Your job is to summarize the news from the text, with emphasis on news around large language models, and the tools that are used to handle them.
    The summary should be as long as possible to preserve the fine-detail, but still be concise enough to be read quickly.
    It also should be written in a way that is easy to understand for a junior data scientist or machine learning engineer.

    {text}

    CONCISE SUMMARY: """

    combine_str = """You are given a summary of news in the space of AI, now you need to combine it together. 
    The summary should be as long as possible to preserve the fine-detail, but still be concise enough to be read in a few minutes.

    {text}

    COMBINATION: """

    llm = BardLLM(conversation_id="")
    MAP_PROMPT = PromptTemplate(template=map_str, input_variables=["text"])
    COMBINE_PROMPT = PromptTemplate(
        template=combine_str, input_variables=["text"]
    )
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        verbose=True,
        token_max=1500,
    )

    output_summary = chain.run(docs)
    return output_summary


def text_to_speech(response) -> dict[str, bytes | int]:
    """Convert text to speech using Bard

    Arguments:
        response {str} -- The text to convert to speech

    Returns:
        str -- The audio version of the text
    """

    from bardapi import BardCookies

    bard = BardCookies(token_from_browser=True)
    return bard.speech(response)


def save_summary(text: str, audio: bytes) -> None:
    """Save the text and audio of the summary to a file

    Arguments:
        text {str} -- The text of the summary
        audio {dict} -- The audio of the summary
    """
    today_str = datetime.today().strftime("%Y-%m-%d")

    with open(f"summaries/audio/summary_{today_str}.ogg", "wb") as f:
        f.write(bytes(audio["audio"]))

    with open(f"summaries/text/summary_{today_str}.txt", "w") as f:
        f.write(text)
