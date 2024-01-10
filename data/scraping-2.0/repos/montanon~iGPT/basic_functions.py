import os
import re

import fitz
import openai
import PyPDF2
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm

with open('.env', 'r') as f:
    KEY = f.read().split('=')[-1].replace('\n', '')
    os.environ["OPEN_API_KEY"] = KEY
openai.api_key = KEY


def process_strings(strings_list, available_tokens):
    string = '\n'.join(strings_list)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=available_tokens, chunk_overlap=0)
    processed_strings = text_splitter.split_text(string)
    return processed_strings


def split_string_at_space(text, max_chars):
    if len(text) <= max_chars:
        return text
    split_index = max_chars
    while text[split_index] != ' ' and split_index > 0:
        split_index -= 1
    if split_index == 0:
        # If there's no space before the max_chars, split at the first space after max_chars
        split_index = text.find(' ', max_chars)
    return text[:split_index], text[split_index + 1:]


def load_counter(file_path='token_usage'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            counter = float(file.read().strip())
    else:
        counter = 0
    return counter


def save_counter(counter, file_path='token_usage'):
    with open(file_path, 'w') as file:
        file.write(str(counter))


def increment_counter(usage, file_path='token_usage'):
    counter = load_counter(file_path=file_path)
    counter += usage
    save_counter(counter, file_path=file_path)
    return counter


def select_pdf_file(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            return os.path.join(folder_path, file)
    return None


def extract_pdf_text(pdf_file_path, max_tokens=2500):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=max_tokens, chunk_overlap=0)
    doc = fitz.open(pdf_file_path)
    text = '\n'.join(
        [' '.join(
            [bl[-3]
                for bl in page.get_text('blocks') if bl[-3].find('<image') == -1]
        ) for page in doc])
    texts = text_splitter.create_documents([text])
    return texts


def summarize_pdf_text(texts, pre_prompt=None, max_tokens=3500, tokens_response=200):
    if isinstance(texts, str):
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_tokens, chunk_overlap=0)
        texts = text_splitter.create_documents([texts])
    llm = OpenAI(temperature=0, openai_api_key=KEY)
    if pre_prompt is not None:
        with open(pre_prompt, 'r') as f:
            prompt = f.read()
        PROMPT = PromptTemplate(template=prompt, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT,
                                     combine_prompt=PROMPT, return_intermediate_steps=True)
    else:
        chain = load_summarize_chain(
            llm, chain_type="map_reduce", return_intermediate_steps=True)
    with get_openai_callback() as cb:
        output_text = chain({"input_documents": texts},
                            return_only_outputs=True)
        increment_counter(cb.total_tokens)
        increment_counter(cb.total_cost, file_path='total_cost')
    paragraphs_summary = '\n'.join(output_text['intermediate_steps'])
    document_summary = output_text['output_text']
    return paragraphs_summary, document_summary, cb.total_cost


def classify_pdf_text(summary):
    with open('pre-prompt_classification.txt', 'r') as f:
        pre_prompt = f.read()
    prompt = f"{pre_prompt}\n{summary}"
    response = openai.Completion.create(
        engine="text-davinci-003",  # "gpt-3.5-turbo"
        prompt=prompt,
        max_tokens=400,
        n=1,
        stop=None,
        temperature=0.15,
    )
    increment_counter(response["usage"]["total_tokens"])
    output_text = response.choices[0].text.strip() + "\n\n"
    return output_text.strip()


def main():
    folder_path = '/Users/sebastian/Desktop/MontagnaInc/TODAI/Yoshida-sensei/References'
    pdf_file_path = os.path.join(
        folder_path, 'A draft human pangenome reference.pdf')
    if pdf_file_path is None:
        print("No PDF file found in the specified folder.")
        return
    paragraphs = extract_pdf_text(pdf_file_path)
    summary = summarize_pdf_text(paragraphs)
    with open('summary.txt', 'w') as f:
        f.write(summary)
    print(f"Summarized PDF content:\n{summary}")
    classification = classify_pdf_text(summary)
    with open('classification.txt', 'w') as f:
        f.write(classification)
    print(f"Classified PDF content:\n{classification}")


def get_summary(pdf_path):
    texts = extract_pdf_text(pdf_path, max_tokens=3750)
    summary = summarize_pdf_text(
        texts, pre_prompt='pre-prompt_summary.txt', max_tokens=4097, tokens_response=200)
    return summary


if __name__ == "__main__":
    # main()
    pdf_path = '/Users/sebastian/Desktop/MontagnaInc/TODAI/Yoshida-sensei/References/A draft human pangenome reference.pdf'
    summary = get_summary(pdf_path)
