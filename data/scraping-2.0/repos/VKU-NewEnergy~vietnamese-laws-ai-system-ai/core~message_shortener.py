"""
Copyright (c) VKU.NewEnergy.

This source code is licensed under the Apache-2.0 license found in the
LICENSE file in the root directory of this source tree.
"""

from llm.base_model.langchain_openai import LangchainOpenAI

from langchain.text_splitter import TokenTextSplitter
from langchain.schema import HumanMessage

def split_text(tokens, chunk_size, overlap_size):
    if len(tokens) <= chunk_size:
        yield tokens
    else:
        chunk = tokens[:chunk_size]
        yield chunk
        yield from split_text(tokens[chunk_size - overlap_size :], chunk_size, overlap_size)


def split_text_to_chunks(text: str, chunk_size=1000, overlap_size=100):
    text_splitter = TokenTextSplitter(
            model_name="gpt-3.5-turbo",
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
        )  
    return text_splitter.split_text(text)

def convert_to_detokenized_text(tokenized_text):
    prompt_text = " ".join(tokenized_text)
    detokenized_text = prompt_text.replace(" 's", "'s")
    return detokenized_text

def shorten_message(message: str, max_words: int = 150) -> str:
    consoloidated_message = ""
    is_chat_model, _, llm_model = LangchainOpenAI.load_llm_model()

    prompt_response = []
    chunks = split_text_to_chunks(message)

    for i, chunk in enumerate(chunks):
        chunk_text = convert_to_detokenized_text(chunk)
        shorten_message_prompt = f"""Given the following message, shorten it in maximum {max_words} words.
        Message: {chunk_text}
        Shortened message:\n\n"""

        if is_chat_model:
            shortened_message = llm_model.generate([[HumanMessage(content=shorten_message_prompt)]])
        else:
            shortened_message = llm_model.generate([shorten_message_prompt])
        if shortened_message:
            shortened_message = shortened_message.generations[0][0].text.strip()
            prompt_response.append(shortened_message)

    if len(prompt_response) == 1:
        consoloidated_message = prompt_response[0]
    elif len(prompt_response) > 1:
        consoloidate_prompt = f"""Consoloidate these messages into a single message in maximum {max_words} words.
        Messages: {str(prompt_response)}
        Consoloidated message:\n\n"""
        if is_chat_model:
            consoloidated_message = llm_model.generate([[HumanMessage(content=consoloidate_prompt)]])
        else:
            consoloidated_message = llm_model.generate([consoloidate_prompt])
        if consoloidated_message:
            consoloidated_message = consoloidated_message.generations[0][0].text.strip()

    return consoloidated_message