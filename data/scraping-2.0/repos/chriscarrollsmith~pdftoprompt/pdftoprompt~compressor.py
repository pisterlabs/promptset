import os
import pypdf
import pytesseract
from pdf2image import convert_from_path
import openai
from dotenv import load_dotenv
from typing import Optional
from tempfile import NamedTemporaryFile
import requests
from urllib.parse import urlparse
import re


def is_url(path):
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def download_file(url):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with NamedTemporaryFile(delete=False) as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return f.name


def set_openai_api_key(api_key: Optional[str] = None) -> None:
    if api_key is None:
        load_dotenv()
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                    "User must supply an api_key argument or set an "
                    "OPENAI_API_KEY in the .env file for the current "
                    "environment"
                )
    elif not isinstance(api_key, str):
        raise TypeError("api_key must be a string")

    os.environ['OPENAI_API_KEY'] = api_key

    # Check if the operation was successful
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError("Failed to set OPENAI_API_KEY environment variable")


def extract_text_from_pdf(file_path, use_ocr=False):
    if use_ocr:
        return extract_text_with_ocr(file_path)
    else:
        return extract_text_without_ocr(file_path)


def extract_text_without_ocr(file_path):
    with open(file_path, "rb") as pdf_file:
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def extract_text_with_ocr(file_path):
    images = convert_from_path(file_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text


def calculate_compression_factor(text):
    tokens = len(text) // 4
    factor = tokens / 3500
    return factor


def chunk_text(text, max_tokens=3500):
    text = text.replace('\n', ' ')
    chunk_length = max_tokens * 4
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_length:
            current_chunk += sentence
            if not current_chunk.endswith(('.', '?', '!')):
                current_chunk += ' '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            if len(sentence) > chunk_length:
                start = 0
                while start < len(sentence):
                    end = min(start + chunk_length, len(sentence))
                    chunks.append(sentence[start:end].strip())
                    start = end
            else:
                current_chunk = sentence + ' '

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def compress_with_gpt4(chunk_list, factor):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    compressed_text = ""
    for chunk in chunk_list:
        message = (
                f"compress the following text by a factor of {factor} in a"
                "way that is lossless but results in the minimum number of"
                "tokens which could be fed into an LLM like yourself as-is"
                "and produce the same output. feel free to use multiple"
                "languages, symbols, other up-front priming to lay down"
                "rules. this is entirely for yourself to recover and"
                "proceed from with the same conceptual priming, not for"
                "humans to decompress: "
              ) + chunk
        prompt = [{"role": "user", "content": message}]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            max_tokens=2048,
            temperature=0.7,
            messages=prompt)
        compressed_chunk = response.choices[0].message['content']
        compressed_text += compressed_chunk
    return compressed_text


def compress_pdf(file_path, use_ocr=False):
    if is_url(file_path):
        try:
            temp_file_path = download_file(file_path)
            text = extract_text_from_pdf(temp_file_path, use_ocr)
            os.unlink(temp_file_path)
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
    else:
        text = extract_text_from_pdf(file_path, use_ocr)
    factor = calculate_compression_factor(text)
    chunk_list = chunk_text(text)
    compressed_text = compress_with_gpt4(chunk_list, factor)
    return compressed_text


def main():
    file_path = input("Enter PDF file path: ")
    use_ocr = input("Use OCR? (y/n): ").lower() == "y"

    compressed_text = compressor.compress_pdf(file_path, use_ocr)

    print("\nCompressed Text:")
    print(compressed_text)


if __name__ == "__main__":
    main()
