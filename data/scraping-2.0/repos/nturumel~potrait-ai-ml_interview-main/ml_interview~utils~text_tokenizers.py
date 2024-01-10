from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from transformers import BertTokenizerFast, GPT2TokenizerFast

from ml_interview.utils.constants import (
    BERT_CHUNK_OVERLAP,
    BERT_CHUNK_SIZE,
    GPT_CHUNK_OVERLAP,
    GPT_CHUNK_SIZE,
    TOKEN_SEPERATORS,
)


def get_bert_text_splitter() -> TextSplitter:
    """
    Initialize a BERT tokenizer and get a text splitter for it.
    """
    bert_tokenizer = load_bert_tokenizer()
    bert_text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        bert_tokenizer,
        chunk_size=BERT_CHUNK_SIZE,
        chunk_overlap=BERT_CHUNK_OVERLAP,
        separators=TOKEN_SEPERATORS,
    )
    return bert_text_splitter


def get_gpt_text_splitter() -> TextSplitter:
    """
    Initialize a GPT-2 tokenizer and get a text splitter for it.
    """
    gpt_tokenizer = load_gpt_tokenizer()
    gpt_text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        gpt_tokenizer,
        chunk_size=GPT_CHUNK_SIZE,
        chunk_overlap=GPT_CHUNK_OVERLAP,
        separators=TOKEN_SEPERATORS,
    )
    return gpt_text_splitter


def load_bert_tokenizer() -> BertTokenizerFast:
    """
    Load and return BERT tokenizer.
    """
    return BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)


def load_gpt_tokenizer() -> GPT2TokenizerFast:
    """
    Load and return GPT-2 tokenizer.
    """
    return GPT2TokenizerFast.from_pretrained("gpt2")


def calculate_bert_length(text: str) -> int:
    """
    Calculate the number of tokens in the given text using BERT tokenizer.
    """
    tokenizer = load_bert_tokenizer()
    return len(tokenizer.encode(text))


def calculate_gpt_length(text: str) -> int:
    """
    Calculate the number of tokens in the given text using GPT-2 tokenizer.
    """
    tokenizer = load_gpt_tokenizer()
    return len(tokenizer.encode(text))
