#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ***************************************************************************80
#
# python ./summarize/summarize_long_transcripts.py
#
# This script summarizes transcripts/conversations that exceeds the 4k token
# context window (prompt + completion).
#
# This script is executed only after executing summarize_transcripts.py script.
#
# This uses gpt-3.5-turbo model.
#
# Accepts
#  - Conversations/Transcripts in HumanFirst Dataformat
#  - Openai API Key
#  - Text file containing ids of large conversations/transcripts
#    (one of the output of summarize_transcripts.py)
#  - Number of cores to use for parallelization (optional)
#  - Number of conversations to summarize (optional)
#  - File path of server log (optional)
#
# Prompt format:
#
# Conversations are split into segments - each segment not exceeding 1800 tokens
# First segment of a conversation is summarized using the folloing prompt format:
#
# """
# <contents of prompt1.txt file>
#
# <first segment of conversation/transcript>
#
# <contents of prompt2.txt file>
# """
#
# Rest of the segments of a conversation is processed using the following prompt
# format:
#
# """
# <contents of prompt3.txt file>
#
# <output of the last openai api call
# - containing the info extracted from the previous segments of the conversation>
#
# <contents of prompt4.txt file>
#
# <current segment of conversation/transcript>
#
# <contents of prompt5.txt file>
# """
#
# Parallelization of API calls helps to summarize large number of transcripts
#
# Saves the summary of each transcript, along with the summaries after
# processing each segment as an individual text file.
#
# *****************************************************************************

# standard imports
import json
from os.path import exists
import random
import logging
from os.path import join
from pathlib import Path
import math
import re
from multiprocessing import Pool
import time
from time import perf_counter

# 3rd party imports
import openai
import pandas
import numpy
import click
import tiktoken
import nltk

START_TIME = perf_counter()

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@click.command()
@click.option('-i', '--input_filepath', type=str, required=True, help='Path containing HF Unlabelled conversations in json format')
@click.option('-a', '--openai_api_key', type=str, required=True, help='OpenAI API key')
@click.option('-n', '--num_cores', type=int, default=8, help='Number of cores for parallelisation')
@click.option('-c', '--conversation_count', type=int, default=100, help='Number of conversations to process')
@click.option('-s', '--server', type=str, default='', help='Server log file path')
@click.option('-l', '--long_convo_ids_file', type=str, required=True, help='long conversation ids text file path')
def main(input_filepath: str, openai_api_key: str, num_cores: int, conversation_count: str, server: str, long_convo_ids_file: str):
    '''Main Function'''
    process(input_filepath, openai_api_key, num_cores, conversation_count, server, long_convo_ids_file)


def process(input_filepath: str, openai_api_key: str, num_cores: int, conversation_count: str, server: str, long_convo_ids_file: str):
    '''Summarization of Conversations'''

    # logging config
    if server == "":
        server = join(str(Path(input_filepath).parent), "server_summarize.log")

    logging.basicConfig(filename=server, filemode='w', level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(process)d - %(levelname)s -- %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    openai.api_key = openai_api_key

    # load input data
    with open(input_filepath, mode="r", encoding="utf8") as f:
        data = json.load(f)
    df = pandas.json_normalize(data=data["examples"], sep="-")

    # set index as conversation ID
    df.set_index(["context-context_id"], drop=True, inplace=True)

    with open(long_convo_ids_file, mode="r", encoding="utf-8") as f:
        long_convo_ids = f.read()

    long_convo_ids = [long_convo_id.strip() for long_convo_id in long_convo_ids.split("\n")]

    long_convo_ids = [long_convo_ids[0]]

    df = df[df.index.isin(long_convo_ids)]
    assert isinstance(df, pandas.DataFrame)

    print(len(df.index.unique()))

    # make one conversations per row so that when splitting,
    # all the converation utterances stays intact
    cols = list(df.columns)
    agg_dict = {}

    for col in cols:
        agg_dict.update({
            col: lambda x: x.tolist()
        })

    df = df.groupby(df.index).agg(agg_dict)

    dfs = numpy.array_split(df, num_cores)
    parallelization_input = []
    for dataframe in dfs:
        dataframe = dataframe.explode(cols)
        parallelization_input.append([dataframe, input_filepath])

    # parallelization
    with Pool(num_cores) as p:
        parallelization_output = p.map(summarization, parallelization_input)

    convo_processed_num = len(df.index.unique().to_list())
    logging.info(f"Total number of conversations processed is {convo_processed_num}")
    logging.info(f'Total Duration for the script: {time.strftime("%H:%M:%S", time.gmtime(perf_counter() - START_TIME))}')


def summarization(input: list) -> None:
    '''Summarization'''

    df = input[0]
    input_filepath = input[1]

    indices = df.index.unique().to_list()
    len_of_indices = len(indices)
    i = 0

    with open("./summarize/prompt1.txt", mode="r", encoding="utf8") as f:
        prompt1 = f.read()

    with open("./summarize/prompt2.txt", mode="r", encoding="utf8") as f:
        prompt2 = f.read()

    with open("./summarize/prompt3.txt", mode="r", encoding="utf8") as f:
        replace_prompt3 = f.read()

    with open("./summarize/prompt4.txt", mode="r", encoding="utf8") as f:
        replace_prompt4 = f.read()

    with open("./summarize/prompt5.txt", mode="r", encoding="utf8") as f:
        replace_prompt5 = f.read()

    replace_list = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                    "eleventh", "twelfth", "thirteen"]

    while i < len_of_indices:

        index = indices[i]

        example_df = df[df.index.isin([index])]

        # get the conversation as client-expert dialogue
        pt = nltk.tokenize.PunktSentenceTokenizer()
        conversation = get_conversation(example_df, pt)

        # conversation segments
        conversation_count = len(conversation)
        if conversation_count <= 13:

            # processing first part of the conversation
            summary_path_first_part = f"{input_filepath.split('.json')[0]}_conversation_{index}_summary_1.txt"
            conversation1 = conversation[0]
            summary = ""
            summarization_success, summary = summarization_helper(index, conversation1, summary_path_first_part, prompt1, prompt2, summary)
            if summarization_success == 0:
                continue

            # processing multiparts of the conversation
            j = 1
            while j < conversation_count:
                if j == 1:
                    part1 = replace_list[0]
                    part2 = replace_list[j]
                    prompt3 = re.sub("<replace>", part1, replace_prompt3)
                    prompt4 = re.sub("<replace>", part2, replace_prompt4)
                    prompt5 = re.sub("<replace>", ", and ".join([part1, part2]), replace_prompt5)

                else:
                    part1 = ", ".join(replace_list[0:j - 1])
                    part2 = replace_list[j - 1]
                    part3 = replace_list[j]
                    prompt3 = re.sub("<replace>", ", and ".join([part1, part2]), replace_prompt3)
                    prompt4 = re.sub("<replace>", part3, replace_prompt4)
                    prompt5 = re.sub("<replace>", ", and ".join([(", ".join([part1, part2])), part3]), replace_prompt5)

                if j == conversation_count - 1:
                    summary_path = f"{input_filepath.split('.json')[0]}_conversation_{index}_summary.txt"
                else:
                    summary_path = f"{input_filepath.split('.json')[0]}_conversation_{index}_summary_{j+1}.txt"

                conversation_part = conversation[j]
                multi_run_prompt = f"{prompt3}\n\n{summary}\n\n{prompt4}"

                summarization_success, summary = summarization_helper(index, conversation_part, summary_path, multi_run_prompt, prompt5, summary)
                if summarization_success == 0:
                    continue
                if summarization_success == 1:
                    j = j + 1
            i = i + 1
        else:
            logging.warning(f"Conversation {index} has more than 13 segments")
            print(f"Conversation {index} has more than 13 segments")


def summarization_helper(index: str, conversation: str, summary_path: str, prompt1: str, prompt2: str, summary: str):

    try:
        if not exists(summary_path):
            logging.warning(f"Summary for conversation ID {index} doesn't exists in the path {summary_path}")
            summary = call_api(index, conversation, summary_path, prompt1, prompt2)

        else:
            logging.info(f"Summary for conversation ID {index} already exists\nReading the summary from file {summary_path}")
            with open(summary_path, mode="r", encoding="utf8") as f:
                summary = f.read()
                if summary == "":
                    logging.warning(f"Summary file {summary_path} is empty for conversation ID {index}")
                    summary = call_api(index, conversation, summary_path, prompt1, prompt2)
        return 1, summary

    except Exception as e:
        logging.error(f"Error upon calling API - {e} - id - {index}")
        print(f"Error upon calling API - {e} - id - {index}")
        sec = 5
        time.sleep(sec)
        logging.info(f"Retrying API call for conversation - {index}")
        return 0, summary


def call_api(index: str, conversation: str, summary_path: str, prompt1: str, prompt2: str) -> str:
    '''Call OpenAI API for summarization'''

    logging.info(f"Calling OpenAI to summarize conversation - {index}")
    summary, total_tokens = summarize(conversation, prompt1, prompt2)
    logging.info(f"Total tokens for conversation id {index} is {total_tokens}")
    logging.info(f"Conversation - {index} is summarized")
    # if summary_path.split(".txt")[0][-1] != "1":
    with open(summary_path, mode="w", encoding="utf8") as f:
        f.write(summary)
    logging.info(f"Summary is saved at {summary_path}")

    return summary


def summarize(text, prompt1, prompt2) -> str:
    '''Summarizes single conversation using prompt'''

    prompt = f"""{prompt1}\n\n```\n{text}\n```\n\n{prompt2}"""

    prompt_token = len(encoding.encode(prompt))
    model_prompt_and_completion_token = 4088  # actual value is 4097 but tiktoken produces only approx value
    completion_token = model_prompt_and_completion_token - prompt_token

    if completion_token >= 600 and completion_token <= 1200:
        completion_token = completion_token
    elif completion_token < 600:
        completion_token = 600
    else:
        completion_token = 1200

    # print(prompt)
    print(prompt_token, completion_token)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=completion_token,
        top_p=1,                # default value
        frequency_penalty=0.0,  # default value
        presence_penalty=0.0    # default value
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content, response.usage.total_tokens


def get_conversation(example_df: pandas.DataFrame, pt: nltk.tokenize.PunktSentenceTokenizer) -> str:
    '''Converts the conversations in HF format to customer-agent dialogue'''

    utterances = []
    for key, row in example_df.iterrows():
        list_sentences = pt.tokenize(row["text"])
        if row["context-role"] == "client":
            for sentence in list_sentences:
                utterances.append(f'Customer: {sentence}')
        else:
            for sentence in list_sentences:
                utterances.append(f'Agent: {sentence}')
    # return "\n".join(utterances)
    num_of_tokens = len(encoding.encode("\n".join(utterances)))
    max_token_for_conversation_part = 1800
    len_of_utterances = len(utterances)

    # number of parts a conversation is split

    num_of_parts = math.ceil(num_of_tokens / max_token_for_conversation_part)
    divide_pt = len_of_utterances // num_of_parts
    print(num_of_tokens, num_of_parts, divide_pt, len_of_utterances)

    conversation = []
    for i in range(num_of_parts):
        if i == num_of_tokens - 1:
            convo = utterances[(divide_pt * i):len_of_utterances]
        else:
            convo = utterances[(divide_pt * i):(divide_pt * (i + 1))]
        conversation.append("\n".join(convo))
    return conversation


if __name__ == "__main__":
    main()
