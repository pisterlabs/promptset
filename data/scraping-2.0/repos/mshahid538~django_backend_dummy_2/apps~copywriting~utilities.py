import time
import asyncio
import openai
import os
import functools
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from chronological import _trimmed_fetch_response, gather
import nltk
from nltk.corpus import wordnet
import requests
from bs4 import BeautifulSoup
import random

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")

DEBUG_MODE = False

# Setup NLTK
file_path = '{}'.format(Path(__file__).resolve().parent.parent)
nltk.data.path.append(file_path)
nltk.download('wordnet', download_dir=file_path)
nltk.download('averaged_perceptron_tagger', download_dir=file_path)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def find_synonym_dic(word):
    """
    Find synonym from thesaurus
    TODO: only first 5 results will be used. need to have a better way to find the exact meaning of the input word
    """
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(word))
    soup = BeautifulSoup(response.text, 'lxml')
    soup.find(
        'section', {'class': 'MainContentContainer css-ln1i60 e1h3b0ep0'})
    result = [span.text for span in soup.findAll('a', {
        'class': 'css-1kg1yv8 eh475bn0'})]  # 'css-1kg1yv8 eh475bn' for less relevant synonyms
    if len(result) == 0:
        return ""
    elif len(result) < 5:
        return random.sample(result, 1)[0]
    else:
        return random.sample(result[:5], 1)[0]


def create_synonyms(sentence):
    """
    Create synonyms for words in a sentence
    """
    word_dic = {}
    text = nltk.word_tokenize(sentence)
    result = nltk.pos_tag(text)
    syn_result = []
    for i in range(len(result)):
        if get_wordnet_pos(result[i][1]) == wordnet.ADJ:
            syn = find_synonym_dic(result[i][0])
            if syn != "":
                syn_result.append(find_synonym_dic(result[i][0]))
            else:
                syn_result.append(result[i][0])
        else:
            syn_result.append(result[i][0])
    new_sentence = " ".join(syn_result)
    logger.debug(f"New sentence: {new_sentence}")
    return new_sentence


def execute(fn, *args):
    """
    Main function that runs logic. Accepts a function implemented on your end!
    Run fn three times in parallel
    """
    tic = time.perf_counter()
    result = asyncio.run(
        gather(fn(*args), fn(*args), fn(*args), fn(*args), fn(*args)))
    toc = time.perf_counter()
    logger.debug(f"FINISHED WORKFLOW IN {toc - tic:0.4f} SECONDS")
    return result


async def autogen_prompt(prompt):
    block_1 = await cleaned_completion(prompt, engine="curie", max_tokens=100, temperature=0.8, top_p=0.5,
                                       presence_penalty=1, frequency_penalty=1, debug_mode=DEBUG_MODE)
    block_1 = block_1.split('"')[0]
    print(f'Completion Response: {block_1}\n')
    return block_1


async def cleaned_completion(prompt, engine="ada", max_tokens=64, temperature=0.7, top_p=1, stop=None,
                             presence_penalty=0,
                             frequency_penalty=0, echo=False, n=1, stream=False, logprobs=None, best_of=1,
                             logit_bias={},
                             debug_mode=False):
    """
       Wrapper for OpenAI API completion. Returns trimmed result from GPT-3.
    """
    if debug_mode:
        logger.debug("""CONFIG:
        Prompt: {0}
        Temperature: {1}
        Engine: {2}
        Max Tokens: {3}
        Top-P: {4}
        Stop: {5}
        Presence Penalty {6}
        Frequency Penalty: {7}
        Echo: {8}
        N: {9}
        Stream: {10}
        Log-Probs: {11}
        Best Of: {12}
        Logit Bias: {13}"""
                     .format(repr(prompt), temperature, engine, max_tokens, top_p, stop, presence_penalty,
                             frequency_penalty, echo, n, stream, logprobs, best_of, logit_bias))
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, functools.partial(openai.Completion.create, engine=engine,
                                                                  prompt=prompt,
                                                                  max_tokens=max_tokens,
                                                                  temperature=temperature,
                                                                  top_p=top_p,
                                                                  presence_penalty=presence_penalty,
                                                                  frequency_penalty=frequency_penalty,
                                                                  echo=echo,
                                                                  stop=stop,
                                                                  n=n,
                                                                  stream=stream,
                                                                  logprobs=logprobs,
                                                                  best_of=best_of,
                                                                  logit_bias=logit_bias))

    if debug_mode:
        logger.debug("GPT-3 Completion Result: {0}".format(response))
    return _trimmed_fetch_response(response, n)
