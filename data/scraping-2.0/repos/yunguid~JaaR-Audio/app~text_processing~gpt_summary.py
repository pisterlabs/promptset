import os
import spacy
import openai
import logging
import textwrap
import tiktoken
from typing import List, Optional


# Configuration
logging.basicConfig(level=logging.DEBUG)
OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
SPACY_MODEL: str = 'en_core_web_sm'
MAX_SENTENCES: int = 5

# api engine should be gpt3.5 turbo
API_ENGINE: str = 'gpt-3.5-turbo-16k'

# is API Key present
if not OPENAI_API_KEY:
    logging.error("Missing OpenAI API Key!")
    exit(1)

# OpenAI API key and Spacy model
openai.api_key = OPENAI_API_KEY
nlp = spacy.load(SPACY_MODEL)

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def add_punctuation_and_correct_grammar(sentence: str) -> str:
    logging.debug(f"Token Count for Punctuation and Grammar Correction: {len(sentence.split())}")

    response = openai.ChatCompletion.create(
        model=API_ENGINE,
        messages=[
            {"role": "system", "content": "You are a grammar correction model."},
            {"role": "user", "content": sentence}
        ],
        max_tokens=60
    )

    logging.debug(f"API Response for Punctuation and Grammar Correction: {response}")

    return response.choices[0].message['content'].strip()

def split_into_paragraphs(text, max_length=1000):
    """Split the text into chunks approximately `max_length` characters long."""
    return textwrap.wrap(text, width=max_length)

def chunk_transcript_by_sentences(transcript: str, max_sentences: int = MAX_SENTENCES) -> List[str]:
    sentences = split_into_sentences(transcript)
    chunks = []
    current_chunk = []
    sentence_count_in_chunk = 0

    for sentence in sentences:
        sentence_count_in_chunk += 1
        if sentence_count_in_chunk <= max_sentences:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            sentence_count_in_chunk = 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def chunk_transcript_by_tokens(transcript: str, max_tokens: int = 4095) -> List[str]:
    sentences = split_into_sentences(transcript)
    chunks = []
    current_chunk = ""
    current_tokens = 0  

    for sentence in sentences: 
        sentence_tokens = count_tokens(sentence)
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk += sentence + " "
            current_tokens += sentence_tokens
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_tokens = sentence_tokens
    chunks.append(current_chunk.strip())
    return chunks


def summarize_transcript(transcript: str) -> str:
    max_tokens = 4095
    response = openai.ChatCompletion.create(
        model=API_ENGINE,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcript},
            {"role": "user", "content": "Can you summarize the key points, context, and any actionable items from the above conversation or video? Please be concise. No talk, just do - Straight to summarizing"}    # Adjusted the prompt
        ],
        max_tokens=max_tokens
    )
    # commented out for less output
    # logging.debug(f"API Response for Summarization: {response}")
    summary = response.choices[0].message['content'].strip()
  
    # Ensures the summary is succinct
    summary = response.choices[0].message['content'].strip()
    summary = summary
    return summary

def process_transcript(transcript: str) -> List[str]:
    # Split final text into paragraphs
    paragraphs = split_into_paragraphs(transcript)
    
    # Generate summaries for each paragraph
    summaries = [summarize_transcript(paragraph) for paragraph in paragraphs]

    return summaries