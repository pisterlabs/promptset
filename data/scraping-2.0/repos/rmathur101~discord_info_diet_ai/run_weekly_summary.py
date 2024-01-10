import sys
from dotenv import load_dotenv
import json
import os
from transformers import GPT2TokenizerFast
from datetime import datetime, date, timedelta
from pathlib import Path
import re
from helpers import get_token_count, format_messages, prompt_summarize_discord_msgs, prompt_remove_no_context_summaries, prompt_consolidate_summaries_where_appropriate, get_today_str, get_one_week_before_ref_date, check_file_exists, check_matching_file_exists_ignore_creation_date, gen_and_get_discord_export
from constants import DISCORD_EXPORT_DIR_PATH, DISCORD_EXPORT_DIR_PATH_RAW, DISCORD_TOKEN_ID, CHANNEL_AND_THREAD_IDS, COMPLETIONS_MODEL
import openai
load_dotenv()

# my prompts 
PROMPTS = {
    "summarize_discord_msgs": prompt_summarize_discord_msgs,
    "remove_no_context_summaries": prompt_remove_no_context_summaries,
    "consolidate_summaries_where_appropriate": prompt_consolidate_summaries_where_appropriate
}

# dict of channel key to message data
channel_key_to_message_data = {}

# gen and load discord exports for all channels
for channel_key in CHANNEL_AND_THREAD_IDS:
    #TODO this should be today's date when you want to generate a weekly summary; in fact you may need to do today's date + 1 - need to check this 
    # reference_date = '2023-04-27'
    # reference_date = '2023-05-08'
    reference_date = '2023-05-15'

    # create discord export (NOTE: json or htmldark for type)
    file_name, file_type = gen_and_get_discord_export(
        DISCORD_EXPORT_DIR_PATH, 
        DISCORD_TOKEN_ID, 
        channel_key, 
        'json',
        get_one_week_before_ref_date(reference_date=reference_date), 
        reference_date, 
        False
    )

    # read file as json if json
    if file_type == 'json':
        with open(DISCORD_EXPORT_DIR_PATH_RAW + '/' + file_name)  as f:
            data = json.load(f)
            channel_key_to_message_data[channel_key] = data
    
    # TODO: calling break will do it for one channel for testing
    break

# exit program
# exit()

# TODO: need to fix this so that I'm maxing out max tokens every time, and also adjusting these numbers in case I use GPT-4
MAX_PROMPT_TOKENS = 2500
# MAX_PROMPT_TOKENS = 6500
COMPLETIONS_API_PARAMS = {
    "model": COMPLETIONS_MODEL,
    "temperature": 0, # We use temperature of 0.0 because it gives the most predictable, factual answer.
    # "top_p": 1,
    # "max_tokens": 1200 
}
summarize_discord_msgs_responses = {}
# iterate through channel key to message data
for channel_key in channel_key_to_message_data:
    summarize_discord_msgs_responses[channel_key] = [] 

    # get message data
    message_data = channel_key_to_message_data[channel_key]
    messages_structured = format_messages(message_data['messages'])

    # this is what we will insert into the prompt    
    insert_discord_msgs_str = ""
    insert_discord_msgs_str_token_count = 0

    # iterate through messages_structured
    for message_structured in messages_structured:
        message_str = message_structured + "\n"
        tokens_count = get_token_count(message_str)

        # if we are over the token limit, we don't want to include the current message, but we want to take the insert_discord_msgs_str and insert it into the prompt and then call the api; then we want to reset the insert str and token count and add the current message to it and continue iterating
        if insert_discord_msgs_str_token_count + tokens_count > MAX_PROMPT_TOKENS:
            prompt = PROMPTS['summarize_discord_msgs'](insert_discord_msgs_str)
            print(f"\nCHANNEL KEY: {channel_key}\nPROMPT:\n{prompt}\n")

            # call api
            response = openai.ChatCompletion.create(messages=[{"role": "user", "content": prompt}], **COMPLETIONS_API_PARAMS)

            summarize_discord_msgs_responses[channel_key].append(response)

            # reset insert_discord_msgs_str and insert_discord_msgs_str_token_count
            insert_discord_msgs_str = ""
            insert_discord_msgs_str_token_count = 0

            # add current message to insert_discord_msgs_str and update token count
            insert_discord_msgs_str += message_str
            insert_discord_msgs_str_token_count += tokens_count
        else:
            # add current message to insert_discord_msgs_str and update token count
            insert_discord_msgs_str += message_str
            insert_discord_msgs_str_token_count += tokens_count

            # if we are at the end of the messages_structured, we want to call the api
            if message_structured == messages_structured[-1]:
                prompt = PROMPTS['summarize_discord_msgs'](insert_discord_msgs_str)
                print(f"\nCHANNEL KEY: {channel_key}\nPROMPT:\n{prompt}\n")

                # call api
                response = openai.ChatCompletion.create(messages=[{"role": "user", "content": prompt}], **COMPLETIONS_API_PARAMS)
                summarize_discord_msgs_responses[channel_key].append(response)

                # reset insert_discord_msgs_str and insert_discord_msgs_str_token_count
                insert_discord_msgs_str = ""
                insert_discord_msgs_str_token_count = 0

print(f"\nSUMMARIZE DISCORD MSGS RESPONSES:\n{summarize_discord_msgs_responses}\n")

# concat summaries in summarize_discord_msgs_responses, remove no context, and consolidate where appropriate
for channel_key in summarize_discord_msgs_responses:

   # get responses for channel
    responses = summarize_discord_msgs_responses[channel_key]

    # get summaries from responses and create concat summaries str
    concat_summaries_str = ""
    for response in responses:
        summary_str = response.choices[0].message.content
        concat_summaries_str += summary_str + "\n"

    # remove summaries that don't have context
    prompt = PROMPTS['remove_no_context_summaries'](concat_summaries_str)
    print(f"\nCHANNEL KEY: {channel_key}\nPROMPT: remove_no_context_summaries\n{prompt}\n")
    response = openai.ChatCompletion.create(messages=[{"role": "user", "content": prompt}], **COMPLETIONS_API_PARAMS)
    print(f"\nRESPONSE: remove_no_context_summaries\n{response.choices[0].message.content}\n")

    no_context_summaries_str = response.choices[0].message.content

    # consolidate summaries where appropriate
    prompt = PROMPTS['consolidate_summaries_where_appropriate'](no_context_summaries_str)
    print(f"\nCHANNEL KEY: {channel_key}\nPROMPT: consolidate_summaries_where_appropriate\n{prompt}\n")
    response = openai.ChatCompletion.create(messages=[{"role": "user", "content": prompt}], **COMPLETIONS_API_PARAMS)
    print(f"\nRESPONSE: consolidate_summaries_where_appropriate\n{response.choices[0].message.content}\n")


