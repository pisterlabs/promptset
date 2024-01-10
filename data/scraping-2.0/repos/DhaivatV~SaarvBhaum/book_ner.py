import openai
from dotenv import load_dotenv
from dotenv import dotenv_values
import pandas as pd
import fitz
import urllib.request
import re
import random
import string

load_dotenv()
config = dotenv_values(".env")

openai.api_key = config['openai_api_key']

def generate_random_string(length):

    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string


def download_pdf(url, name):

    output_path = f'D:\\bookgpt\\{name}.pdf'
    urllib.request.urlretrieve(url, output_path)
    return output_path


def preprocess(text):

    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):

    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):

    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[{idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


def generate_text(prompt, engine="text-davinci-003"):
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message


def generate_answer(merged_chunks):


    for chunk in merged_chunks[2:]:
        prompt = ""
        prompt += 'Book Data:\n\n'
        prompt += chunk + '\n\n'

        prompt += "Instructions: Analyze the given data from the book and provide insightful content that can be used to generate trending tweets."\
                "dont Cite each reference using [number] notation (every result has this number at the beginning)."\
                "no Citation should be done at the end of each sentence. If the search results mention multiple subjects"\
                "Make sure the answer is correct and strictly belong to data given from the book don't output false content."\
                "Directly start the answer.\n"

        answer = generate_text(prompt)

        content_dic = {
            "chunk": chunk,
            "content":answer
        }

        df = pd.read_csv("chunk-content.csv")
        df = pd.concat([df, pd.DataFrame([content_dic])], ignore_index=True)
        df.to_csv("chunk-content.csv", index=False)
        print(f"last chunk processed ============ {merged_chunks.index(chunk)}")

    return ("Content Generation Successfull")

def create_chunks_content(url):
    name = generate_random_string(9)
    pdf_path = download_pdf(url,name)
    text = pdf_to_text(pdf_path)
    text = " ".join(text)

    chunks = text_to_chunks(text)
    merged_chunks= []

    for i in range(816, len(chunks), ):

        merged_chunk = ' '.join(chunks[i:i+3])
        merged_chunks.append(merged_chunk)

    return (generate_answer(merged_chunks))



create_chunks_content('https://storagedv.s3.ap-south-1.amazonaws.com/india-that-is-bharat-9789354350047_compress.pdf')