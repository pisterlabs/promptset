import base64
import io
from PyPDF2 import PdfReader
import openai
from pprint import pprint
from text_generation.preprocessing import replace_ligatures, remove_duplicates, remove_close_periods, remove_hyphens, to_lowercase, remove_newlines, split_into_sentences, group_sentences
from text_generation.postprocessing import process_tf, scramble_answers
from text_generation.youtube_processer import youtube_process
import time
from concurrent.futures import ThreadPoolExecutor
import json
import markdown
from bs4 import BeautifulSoup
import requests
from functools import partial
from env import openai_api_key

start = time.time()

GPT_KEY = openai_api_key
openai.api_key = GPT_KEY
TYPE = "website"  # one of: "pdf", "md", "website", "youtube"
PREPROMPT_MC = """I will give you a paragraph. You will create a JSON of 1 multiple-choice question and options from the information in the paragraph.
They must be output in the following format:
{ "question": <question here>, "options": [<answer 1>, <answer 2>, <answer 3>, <answer 4>],  "answer": [<index of correct answer>]}


The paragraph to be sampled from is as follows:

"""
PREPROMPT_TF = """I will give you a paragraph. You will create a JSON of 1 true-false statement and options from the information in the paragraph.
They must be output in the following format:
{ "question": <statement here>, "options": ["true", "false"],  "answer": [<index of correct answer>]}


The paragraph to be sampled from is as follows:

"""
PREPROMPT_SUMMARY = """I will give you a paragraph. You will create a one-sentence summary from the information in the paragraph. Format your output into a single string."""


def summary_prompt(index, text, SUMMARY):
    messages = [
        {"role": "user", "content": PREPROMPT_SUMMARY + text[index] + "\n\n", }]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=150,
        temperature=1.0,
        messages=messages)
    response_text = response.choices[0].message.content
    SUMMARY.append(response_text)


def gpt_prompt(index, text, ALL_QUESTIONS):
    if index % 2 == 0:
        PREPROMPT = PREPROMPT_MC
    else:
        PREPROMPT = PREPROMPT_TF
    messages = [
        {"role": "user", "content": PREPROMPT + text[index] + "\n\n", }]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=150,
        temperature=1.2,
        messages=messages)

    response_json = json.loads(response.choices[0].message.content)


    if index % 2 == 0:
        response_json = scramble_answers(response_json)
        response_json["format"] = "MC"
    else:
        response_json = process_tf(response_json)
        response_json["format"] = "TF"


    ALL_QUESTIONS.append(response_json)


def read_pdf(b64string: str) -> str:
    buffer=base64.b64decode(b64string)
    f=io.BytesIO(buffer)
    reader = PdfReader(f)

    for i, page in enumerate(reader.pages):
        if i == 0:
            text = page.extract_text()
        else:
            text = text + page.extract_text()

    return text


def pdf_preprocess(text: str, SENTENCES_PER_PROMPT) -> list[str]:
    processed_text = split_into_sentences(replace_ligatures(text))
    grouped_text = group_sentences(processed_text, SENTENCES_PER_PROMPT)
    return grouped_text


def read_markdown(utf8_encoded_content: str) -> str:
    markdown_content = utf8_encoded_content
    html_content = markdown.markdown(markdown_content)
    return html_content


def markdown_preprocess(text: list[str], SENTENCES_PER_PROMPT) -> list[str]:
    text = [line for line in text if line != '']
    text = remove_duplicates(text)
    text = group_sentences(text, SENTENCES_PER_PROMPT)
    return text


def process_html(html_content, type):
    soup = BeautifulSoup(html_content, 'html.parser')
    if type == "website":
        paragraphs = soup.find_all('p')
    elif type == "md":
        paragraphs = soup.find_all(['p', 'li'])
    extracted_text = '\n'.join([p.get_text() for p in paragraphs])
    return extracted_text


# type is either "pdf", "md", or "website"
def get_grouped_text(path: str, type: str, SENTENCES_PER_PROMPT) -> list[str]:
    print(f"type: '{type}'")
    if type == "pdf":
        text = read_pdf(path)
        grouped_text = pdf_preprocess(text, SENTENCES_PER_PROMPT)
    elif type == "md":
        markdown_content = read_markdown(path)
        text = process_html(markdown_content, "md").splitlines()
        grouped_text = markdown_preprocess(text, SENTENCES_PER_PROMPT)
    elif type == "website":
        page = requests.get(path)
        text = process_html(page.content, "website").splitlines()
        grouped_text = markdown_preprocess(text, SENTENCES_PER_PROMPT)
    elif type == "youtube":
        text = youtube_process(path)
        text = remove_close_periods(text)
        text = remove_close_periods(text)
        text = remove_newlines(text)
        grouped_text = group_sentences(split_into_sentences(text), SENTENCES_PER_PROMPT)

    return grouped_text


def get_summary(grouped_text) -> list[str]:
    SUMMARY = []
    text_range = list(range(0, len(grouped_text)))
    with ThreadPoolExecutor() as executor:
        partial_func = partial(
            summary_prompt, text=grouped_text, SUMMARY=SUMMARY)
        results = executor.map(partial_func, text_range)

    SUMMARY = ' '.join(SUMMARY)
    return SUMMARY


def get_questions(grouped_text) -> list[dict]:
    ALL_QUESTIONS = []
    text_range = list(range(0, len(grouped_text)))
    with ThreadPoolExecutor() as executor:
        partial_func = partial(gpt_prompt, text=grouped_text, ALL_QUESTIONS=ALL_QUESTIONS)
        results = executor.map(partial_func, text_range)

    print("Question count: ")
    print(len(ALL_QUESTIONS))
    return ALL_QUESTIONS
