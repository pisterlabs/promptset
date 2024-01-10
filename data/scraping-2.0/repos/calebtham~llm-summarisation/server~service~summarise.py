import html
import re
from flask import jsonify
import tiktoken
import langchain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter


DEFAULT_MODEL = "gpt-3.5-turbo-1106"
MAX_TOKENS = 15000

def summarise(text):
    '''Summarise text using GPT-3.5'''

    text = clean_text(text)
    no_tokens = num_tokens_from_string(text)

    if no_tokens > MAX_TOKENS:
        return map_reduce_summarise(text)
    else:
        model = DEFAULT_MODEL
        return stuff_summarise(text, model)


def num_tokens_from_string(string: str) -> int:
    '''Returns the number of tokens in a text string.'''

    encoding = tiktoken.encoding_for_model(DEFAULT_MODEL)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def clean_text(text):
    '''Remove HTML tags and revert HTML escaped special characters to shorten token length'''

    # remove HTML tags
    text = re.sub("<[^<]*>", "", text)  # This is faster than Beautiful Soup

    # revert HTML escaped special characters
    text = html.unescape(text)

    text = text.replace(u'\xa0', u' ') \
        .replace("\\n", "\n") \
        .replace("\\t", "\t") \
        .replace("\\r", "\r") \
        .replace("\\", "")

    return text


def log_chain_run(summary_chain, text):
    '''Runs a chain and logs the callback'''

    with get_openai_callback() as cb:
        summary = summary_chain(text)
        print(cb)

    return summary


def get_summary_chain(llm):
    '''Returns a summarise chain'''

    summary_chain = LLMChain(
        llm=llm,
        prompt=get_summary_prompt(),
        output_key="summary"
    )

    return summary_chain


def get_summary_prompt():
    '''Returns a summarise prompt'''

    summary_prompt_template = (
        "INSTRUCTION: "
        "- Summarize the following text concisely in bullet point form."
        "- Extract the most important, interesting, and relevant content."
        "- The text was taken from a website, so there may be redundant website information that should be avoided."
        "- Do not state \"The article says\" or \"The text mentions\" or \"The text contains\" or similar statements about the form of the article/text/website in the bullet points."
        "- Do not repeat the same information in multiple bullet points."
        "TEXT:{text}"
        "BULLET POINT SUMMARY:"
    )

    summary_prompt = PromptTemplate.from_template(summary_prompt_template)

    return summary_prompt


def stuff_summarise(text, model):
    '''Summarise text using stuffing method'''

    llm = ChatOpenAI(temperature=0, model_name=model)

    summary_chain = get_summary_chain(llm)

    return log_chain_run(summary_chain, text)["summary"]


def map_reduce_summarise(text):
    '''Summarise text using map reduce method'''

    llm = ChatOpenAI(temperature=0, model_name=DEFAULT_MODEL)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        length_function=num_tokens_from_string,
        chunk_size=MAX_TOKENS,
        chunk_overlap=10,
        add_start_index=True,
    )

    docs = text_splitter.create_documents([text])
    print(f"Number of docs: {len(docs)}")

    # Map
    map_prompt_template = (
        "INSTRUCTION: "
        "- Summarize the following text concisely in bullet point form."
        "- Extract the most important, interesting, and relevant content."
        "- The text was taken from a website, so there may be redundant website information that should be avoided."
        "- Do not state \"The article says\" or \"The text mentions\" or \"The text contains\" or similar statements about the form of the article/text/website in the bullet points."
        "TEXT:{text}"
        "BULLET POINT SUMMARY:"
    )

    map_custom_prompt = PromptTemplate(
        input_variables=['text'],
        template=map_prompt_template
    )

    # Combine
    combine_custom_prompt = get_summary_prompt()

    # MapReduce
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_custom_prompt,
        combine_prompt=combine_custom_prompt,
        verbose=False,
        output_key="summary"
    )

    return log_chain_run(summary_chain, docs)["summary"]


def summarise_self_reflect(text):
    '''Summarise text and check summary using self reflection method'''

    llm = ChatOpenAI(temperature=0, model_name=DEFAULT_MODEL)

    summary_chain = get_summary_chain(llm)

    # Self reflection chain
    self_reflect_prompt_template = (
        "INSTRUCTION: "
        "- Given the original text and a generated summary, refine the summary so that only the correct information is included."
        "- Correct information is information that is stated in the original text."
        "- If all information in the summary is correct, then do not change the summary."
        "- Output should be in bullet point form."
        "TEXT:{text}"
        "SUMMARY:{summary}"
        "NEW BULLET POINT SUMMARY:"
    )

    self_reflect_prompt = PromptTemplate.from_template(
        self_reflect_prompt_template)

    self_reflect_chain = LLMChain(
        llm=llm,
        prompt=self_reflect_prompt,
        output_key="self_reflection_summary"
    )

    # Combine chains
    overall_chain = SequentialChain(
        chains=[summary_chain, self_reflect_chain],
        input_variables=["text"],
        output_variables=["summary", "self_reflection_summary"],
        verbose=True
    )

    summary = log_chain_run(overall_chain, text)

    print(summary["summary"])

    return summary["self_reflection_summary"]
