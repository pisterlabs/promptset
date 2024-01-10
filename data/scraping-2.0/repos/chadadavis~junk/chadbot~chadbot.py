#!/usr/bin/env python

# TODO read:
# https://platform.openai.com/docs/guides/fine-tuning/create-a-fine-tuned-model
# https://help.openai.com/en/articles/5528730-fine-tuning-a-classifier-to-improve-truthfulness
# https://docs.google.com/document/d/1rqj7dkuvl7Byd5KQPUJRxc19BJt8wo0yHNwK84KfU3Q/edit
# https://platform.openai.com/docs/quickstart
# https://platform.openai.com/docs/guides/gpt-best-practices

# 2023-11 See the new GPTs
# https://openai.com/blog/introducing-gpts
# https://chat.openai.com/gpts/editor
# And the Assistants API
# https://platform.openai.com/docs/assistants/overview
# https://platform.openai.com/playground

# API docs
# https://help.openai.com/en/collections/3675931-openai-api
# https://github.com/openai/openai-cookbook/

# Note, no fine-tuning on GPT-4 using the API:
# https://help.openai.com/en/articles/7127982-can-i-fine-tune-on-gpt-4
# Fine-tuning for GPT-3.5 Turbo is now available, with fine-tuning for GPT-4 coming this fall. (2023)
# https://platform.openai.com/finetune

# Alternatively:
# https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models

# Do I need to worry about this:

# To preserve the default model's safety features
# through the  fine-tuning process, fine-tuning training data is passed through
# our Moderation API and a GPT-4 powered moderation system to detect unsafe
# training data that conflict with our safety standards.

def backlog(): ...

# TODO / backlog

# Is fine-tuning a solution to my use case?

# We recommend starting with 50 well-crafted demonstrations and seeing if the
# model shows signs of improvement after fine-tuning. In some cases that may be
# sufficient, but even if the model is not yet production quality, clear
# improvements are a good sign that providing more data will continue to improve
# the model. No improvement suggests that you may need to rethink how to set up
# the task for the model or restructure the data before scaling beyond a limited
# example set.

# So, hand-pick some key converstaion examples from the dataset, and see if I can detect improvement subjectively

# The text exports from WhatsApp lose the references to the message being replied to.
# Lookup: is there any way to preserve that in the export (any other export formats?)
# So, the context of the previous thread is broken without any replies being linked to the replied-to message.
# I could just use timestamps to demarcate a new context/conversation, eg 60+ mins apart
# Or keep the thread/conversation "open" if the previous one ended with/contains a '?'
# Count what my context length (median) is with those assumptions (and assume 4 chars-per-token). Should be < 8K (~32K chars)

# Group conversation threads into a list of messages, and start a new set of messages when it's a new thread.
# (That might also make it easier to split the list of threads/conversations into training/test sets)
# You could just have one list of messages, but I feel like it makes sense to break these into time-blocked conversation threads, since the messages in a thread are part of the same context.
# Each training example is limited to 4096 tokens (I guess that's all the messages in the thread)

# I'm assuming that I use the 'user' and 'assistant' roles for the two people in the conversation ...

# https://en.wikipedia.org/wiki/JSON_streaming#Newline-Delimited_JSON
# {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
# {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
# {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters."}]}

# https://help.openai.com/en/articles/6811186-how-do-i-format-my-fine-tuning-data

# Test the concept here first:
# https://platform.openai.com/playground?mode=chat&model=gpt-4

import re
import sys
from pprint import pprint

import openai
import tiktoken


def pvars(_extra:dict={}):
    """If you're inside a def, call this as: pvars(vars())"""
    _vars = { **globals(), **locals(), **_extra }
    pprint([ [k,_vars[k]] for k in _vars if re.match(r'[a-z]', k)])


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages.

    https://openai.com/pricing # Special pricing for fine-tuning
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    if num_tokens > 4096:
        sys.stderr.write("Each training example is limited to 4096 tokens.\n")
    return num_tokens



# New lines in jsonl: (in strings those must be escaped as \r and \n, respectively)
messages_line = [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]
tokens_n=num_tokens_from_messages(messages_line)

# Once you have the data validated, the file needs to be uploaded in order to be used with a fine-tuning jobs:
openai.File.create(
  file=open("mydata.jsonl", "rb"),
  purpose='fine-tune'
)