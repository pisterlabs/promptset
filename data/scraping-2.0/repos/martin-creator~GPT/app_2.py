# Editing Text Using GPT
# Advanced Text Manipulation

import os
import openai


def init_api():
    ''' Load API key from .env file'''
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ["API_KEY"]
    openai.organization = os.environ["ORG_ID"]


init_api()

# response = openai.Edit.create(
#     model="text-davinci-edit-001",
#     instruction="Translate from English to French, Arabic, and Spanish.",
#     input="The cat sat on the mat.",
# )

# Chaining Completions and Edits
# prompt = "The first programming language to be invented was Plankalkül, which was de\
# signed by Konrad Zuse in the 1940s, but not publicly known until 1972 (and not imple\
# mented until 1998). The first widely known and successful high-level programming l\
# anguage was Fortran, developed from 1954 to 1957 by a team of IBM researchers led \
# by John Backus. The success of FORTRAN led to the formation of a committee of scie\
# ntists to develop a universal computer language; the result of their effort was AL\
# GOL 58. Separately, John McCarthy of MIT developed Lisp, the first language with o\
# rigins in academia to be successful. With the success of these initial efforts, prog\
# ramming languages became an active topic of research in the 1960s and beyond\n\nTwee\
# t with hashtags:"

# english_tweet = openai.Completion.create(
#     model="text-davinci-002",
#     prompt=prompt,
#     temperature=0.5,
#     max_tokens=20,
# )

# english_tweet_text = english_tweet["choices"][0]["text"].strip()
# print("English Tweet: ")
# print(english_tweet_text)


# spanish_tweet = openai.Edit.create(
#     model="text-davinci-edit-001",
#     input=english_tweet_text,
#     instruction="Translate to Spanish",
#     temperature=0.5,
# )


# spanish_tweet_text = spanish_tweet["choices"][0]["text"].strip()
# print("Spanish Tweet: ")
# print(spanish_tweet_text)


# Instruction is Required, Input is Optional


# Getting Cryptocurrency Information Based on a User-Defined Schema (Context stuffing)
prompt = """Input: Bitcoin
Output:
BTC was created in 2008, you can learn more about it here: https://bitcoin.org/en/ a\
nd get the latest price here: https://www.coingecko.com/en/coins/bitcoin.
It's all-time high is $64,895.00 and it's all-time low is $67.81.

Input: Ethereum
Output:
ETH was created in 2015, you can learn more about it here: https://ethereum.org/en/ \
and get the latest price here: https://www.coingecko.com/en/coins/ethereum
It's all-time high is $4,379.00 and it's all-time low is $0.43.

Input: Dogecoin
Output:
DOGE was created in 2013, you can learn more about it here: https://dogecoin.com/ an\
d get the latest price here: https://www.coingecko.com/en/coins/dogecoin
It's all-time high is $0.73 and it's all-time low is $0.000002.

Input: Cardano
Output:\n"""

result = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    max_tokens=200,
    temperature=0,
)


print(result["choices"][0]["text"].strip())
