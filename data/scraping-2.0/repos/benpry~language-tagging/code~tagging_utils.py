import asyncio
from prompts import prompt_templates
from itertools import product
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain import LLMChain
from collections import defaultdict
import pandas as pd


async def async_tag(llm, prompt, sentence):
    if isinstance(prompt, ChatPromptTemplate):
        chain = LLMChain(llm, prompt)
        resp = await chain.arun(sentence)
        return resp
    elif isinstance(llm, ChatOpenAI):
        formatted_prompt = [HumanMessage(content=prompt.format(sentence=sentence))]
        resp = await llm.agenerate([formatted_prompt])
    else:
        formatted_prompt = prompt.format(sentence=sentence)
        resp = await llm.agenerate(formatted_prompt)
    return resp.generations[0][0].text.strip()


def sync_tag(llm, prompt, sentence):
    """
    Tag everything synchronously.
    """
    if isinstance(prompt, ChatPromptTemplate):
        chain = LLMChain(llm, prompt)
        resp = chain.run(sentence)
        return resp
    elif isinstance(llm, ChatOpenAI):
        formatted_prompt = [HumanMessage(content=prompt.format(sentence=sentence))]
        resp = llm.generate([formatted_prompt])
    else:
        formatted_prompt = prompt.format(sentence=sentence)
        resp = llm.generate(formatted_prompt)
    return resp.generations[0][0].text.strip()


async def async_tag_all(llm, prompt_types, sentences):
    prompt_sentences = list(product(prompt_types, sentences))
    tags = [
        async_tag(llm, prompt_templates[prompt_type], sentence)
        for prompt_type, sentence in prompt_sentences
    ]
    tags = await asyncio.gather(*tags)

    sentences_with_tags = defaultdict(dict)
    for i, prompt_sentence in enumerate(prompt_sentences):
        prompt_type, sentence = prompt_sentence
        sentences_with_tags[sentence][prompt_type] = tags[i]

    rows = []
    for sentence, tags in sentences_with_tags.items():
        rows.append({"sentence": sentence, **tags})

    return pd.DataFrame(rows)


def sync_tag_all(model_name, prompt_types, sentences):
    if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
        llm = ChatOpenAI(model_name=model_name, max_retries=15, temperature=0)
    else:
        llm = OpenAI(model_name=model_name)

    prompt_sentences = list(product(prompt_types, sentences))

    tags = [
        sync_tag(llm, prompt_templates[prompt_type], sentence)
        for prompt_type, sentence in prompt_sentences
    ]
    sentences_with_tags = defaultdict(dict)

    for i, prompt_sentence in enumerate(prompt_sentences):
        prompt_type, sentence = prompt_sentence
        sentences_with_tags[sentence][prompt_type] = tags[i]

    rows = []
    for sentence, tags in sentences_with_tags.items():
        rows.append({"sentence": sentence, **tags})

    return pd.DataFrame(rows)


def tag_all(model_name, prompt_types, sentences):
    if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
        llm = ChatOpenAI(model_name=model_name, max_retries=15, temperature=0)
    else:
        llm = OpenAI(model_name=model_name)
    return asyncio.run(async_tag_all(llm, prompt_types, sentences))


if __name__ == "__main__":
    llm = ChatOpenAI(model_name="code-davinci-002", temperature=0)

    prompt_types = list(prompt_templates.keys())
    sentences = ["Hello my name is Ben.", "Try not to die."]
    tagged_sentences = asyncio.run(tag_all(llm, prompt_types, sentences))

    print(tagged_sentences)
