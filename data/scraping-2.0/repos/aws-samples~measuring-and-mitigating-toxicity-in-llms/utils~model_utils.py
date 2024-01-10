#!/usr/bin/env python3

import better_profanity
import random
import os
import torch
from IPython.display import Markdown
from langchain.schema import Document
from langchain import PromptTemplate
import gc
import torch
from datasets import load_from_disk
from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline
import copy
import json
import transformers

transformers.logging.set_verbosity_error()


def _get_summarization_pipeline():
    """
    Get a summarization pipeline using the Google Flan T5 large model.

    Returns:
        pipeline: A pipeline configured for text summarization using the Flan T5 large model.
    """
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large",
        skip_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        use_fast=True,
    )

    # load the model
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-large",
        device_map={"": 0},  # this will load the model in GPU
        torch_dtype=torch.float32,
        return_dict=True,
        load_in_4bit=True,
    )

    # set up pipeline
    flan_pipeline = pipeline(
        model=model,
        task="summarization",
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        tokenizer=tokenizer,
        num_beams=4,
        min_length=50,
        max_length=150,
        length_penalty=2.0,
        repetition_penalty=2.0,
    )
    return pipeline


def _my_llm_api(prompt: str, **kwargs) -> str:
    """
    Generate a custom API response adhering to Guardrail.ai requirements.

    Args:
        prompt (str): The input text prompt for summarization.
        **kwargs: Additional keyword arguments for customization.

    Returns:
        json_text (str): JSON-formatted string containing the generated summary based on the input prompt.
    """
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large",
        skip_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        use_fast=True,
    )

    # load the model
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-large",
        device_map={"": 0},  # this will load the model in GPU
        torch_dtype=torch.float32,
        return_dict=True,
        load_in_4bit=True,
    )

    # set up pipeline
    flan_pipeline = pipeline(
        model=model,
        task="summarization",
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        tokenizer=tokenizer,
        do_sample=False,
        num_beams=4,
        min_length=50,
        max_length=150,
        length_penalty=2.0,
        repetition_penalty=2.0,
    )

    # this needs to match the name tag in the RAIL string
    dict_text = {"summarize_statement": flan_pipeline(prompt)[0]["summary_text"]}
    json_text = json.dumps(dict_text)

    del tokenizer, model, flan_pipeline
    torch.cuda.empty_cache()
    gc.collect()

    return json_text


def _format_llm_output(text):
    """
    Apply formatting to the output from Language Models (LLMs).

    Args:
        text (str): The text to be formatted.

    Returns:
        Markdown: A formatted Markdown object encapsulating the input text.
    """
    return Markdown('<div class="alert alert-block alert-info">{}</div>'.format(text))


def _generate_summary(prompt, model, tokenizer):
    """
    Generate a summary using the specified language model.

    Args:
        prompt (str): The input text prompt for generating the summary.
        model: The language model to be used for text generation.
        tokenizer: The tokenizer associated with the language model.

    Returns:
        output_text (str): The generated summary text.
    """
    # encode text (tokenize)
    encoded_tokens = tokenizer(prompt, return_tensors="pt", truncation=True)

    # generate summary
    generated_tokens = model.generate(
        encoded_tokens.input_ids.to("cuda"),
        num_return_sequences=1,
        do_sample=False,
        early_stopping=True,
        num_beams=4,
        min_length=50,
        max_length=350,
        length_penalty=2.0,
        repetition_penalty=2.0,
    )

    # garbage collect
    del encoded_tokens
    torch.cuda.empty_cache()

    # convert back
    output_text = tokenizer.decode(generated_tokens.reshape(-1))

    # garbage collect
    del generated_tokens
    torch.cuda.empty_cache()

    return output_text


def _add_summaries(sample, chain):
    """
    Add summaries to the movie dialogue dataset using a language model chain.

    Args:
        sample (Dataset): A dictionary representing a sample from the dataset.
        chain: A LangChain instance for processing and summarization.

    Returns:
        sample (Dataset): The sample with an added "summary" key containing the generated summary.
    """
    # turn off verbosity for chain
    chain.llm_chain.verbose = False

    # create LangChain document from the chunks
    docs = [
        Document(page_content=split["text"], metadata=split["metadata"])
        for split in sample["chunks"]
    ]

    # parse documents through the map reduce chain
    full_output = chain({"input_documents": docs})

    # extract the summary
    summary = full_output["output_text"]

    # return the new column
    sample["summary"] = summary

    # delete objects that are no longer in use
    del docs, summary

    # garbage collect
    gc.collect()

    return sample
