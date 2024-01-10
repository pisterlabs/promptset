#  Author Ene SS Rawa / Tjitse van der Molen
import math
import time

import openai
import backoff
import tiktoken

from logger.embedding_logger import logger


@backoff.on_exception(backoff.expo, openai.error.OpenAIError) # cover all errors (RateLimitError, APIError, Timeout etc.)
def text_completions_with_backoff(**kwargs):
    """
	Calls GPT ChatCompletion with exponential backoff

	Input: openai.Chatcompletion API call in same format as under normal conditions
	Output: openai.Chatcompletion API output in same format as under normal conditions
	"""
    return openai.ChatCompletion.create(**kwargs)


# # #

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def audio_transcription_with_backoff(audio_file):
    """
	Calls GPT audio transcription with exponential backoff

	Input: opened audiofile of any of the following formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
	Output: dictionary with "Text" as key and the transcribed text from the audio file as value
	"""
    return openai.Audio.transcribe("whisper-1", audio_file)


# # #

def call_GPT(gpt_prompt, gpt_role="You are a helpful assistant.", start_seq="\n\nSummary:\n",
             include_start_seq_out=False, max_output_tokens=500, temp=0):
    """
	Calls GPT API and returns results

	Input:
	gpt_prompt - str: prompt for GPT call
	start_seq - str: start_inject text. forced start of GPT response
		(default = "\n\nSummary:\n")
	include_start_seq_out - bool: whether start_seq should be included
		in the output (default = False)
	max_output_tokens - int: maximum number of tokens in output (default
		= 500). If a value larger than 3900 or a negative values is
		provided, max_output_tokens will be set to default.
	temp - float: temperature value for GPT call in range from 0 to 1
		(default = 0). If a value outside of the allowed range is
		provided, temp will be set to default.

	Output:
	result - str: GPT response to gpt_prompt
	"""

    # check if max_output_tokens value is within range
    if max_output_tokens < 0 or max_output_tokens > 3900:
        print(
            "ERROR: use max_output_tokens value between 0 and 3900, default value of max_output_tokens=500 will be used")
        max_output_tokens = 500

    # check if temp value is within range
    if temp < 0 or temp > 1:
        print("ERROR: use temp value between 0 and 1, default value of temp=0 will be used")
        temp = 0

    # add start text to prompt
    prompt_with_start = gpt_prompt + start_seq
    logger.debug(f"Waiting some secs")
    time.sleep(1)
    logger.debug(f"Prompt with start: {prompt_with_start}")

    # run gpt prompt through API
    response = text_completions_with_backoff(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": gpt_role}, {"role": "user", "content": prompt_with_start}],
        temperature=temp,
        max_tokens=max_output_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # obtain result output
    if include_start_seq_out:
        result = start_seq + response["choices"][0]["message"]["content"]
    else:
        result = response["choices"][0]["message"]["content"]

    # obtain number of tokens used
    num_tokens_used = response["usage"]["total_tokens"]
    logger.debug(f"=================")
    logger.debug(f"Number of tokens used: {num_tokens_used}")
    logger.debug(f"Result: {result}")
    logger.debug(f"=================")
    return result, num_tokens_used


# # #

def split_prompt_tokens(gpt_prompt, max_tokens, summarize_prompt_basis=None, sep_char=["\n", " "]):
    """
	Counts tokens in prompt and splits it if tokens exceed max_tokens

	Input:
	gpt_prompt - str: prompt text for gpt
	tokenizer - GPT2Tokenizer: tokenized gpt prompt
	max_tokens - int: maximum number of tokens allowed in single prompt
	summarize_prompt_basis - str: prompt basis for gpt (prompt without
		text to be summarized) to be included in prompts after splitting

	Output:
	prompt_list - [str]: list containing split prompts of at most
		max_tokens tokens each

	Notes:
	Prompt splitter is not implemented yet
	"""

    # intitialize encoding
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # encode input text
    token_list = encoding.encode(gpt_prompt)

    # # # SPLIT DATA # # #

    # if the prompt has more tokens than allowed
    if len(token_list) > max_tokens:

        # determine minimal number of splits and add 1
        # num_split = (len(token_list) // max_tokens + (len(token_list) % max_tokens > 0)) + 1
        num_split = math.ceil(len(token_list) / max_tokens)

        # split text with newline separator
        prompt_list = split_gpt_prompt(gpt_prompt, sep_char[0], round(len(gpt_prompt) / num_split),
                                       summarize_prompt_basis)

        # if text could not be split
        if not prompt_list:

            # split text with space separator
            prompt_list = split_gpt_prompt(gpt_prompt, sep_char[1], round(len(gpt_prompt) / num_split),
                                           summarize_prompt_basis)

            # if text could not be split
            if not prompt_list:

                # print error message
                print(
                    "ERROR: Text separators in sep_char do not split text into sufficiently small chunks, adjust text input")

                return False

            else:

                # print warning message
                print(
                    "WARNING: The first separator in sep_char does not split text into sufficiently small chunks, the second separator in sep_char is used instead. This might lead to lower performance")

    else:

        # make single entry list of gpt_prompt
        prompt_list = [gpt_prompt]

    return prompt_list


# # #

def split_gpt_prompt(text, split_string, max_char, prompt_basis):
    """
	Splits a long GPT prompt into smaller equally sized prompts

	Input:
	prompt - str: A GPT prompt to split
	max_char - int: The maximum number of characters for each prompt

	Output:
	prompts - [str]: A list of smaller GPT prompts
	"""

    # split the text based on the provided split_string
    split_text = text.split(split_string)

    # make empty result arrays
    curr_prompt = ""
    all_prompts = []

    # for each split text
    for sp_t in split_text:

        # if the split text is longer than max_char
        if len(sp_t) + len(prompt_basis) > max_char:
            return False

        # if the current prompt plus the added text is smaller than the maximum characters
        if len(curr_prompt) + len(sp_t) <= max_char:

            # add the added text to the current prompt
            curr_prompt = curr_prompt + sp_t + split_string

        else:

            # store current prompt in all prompts
            all_prompts.append(curr_prompt)

            # overwrite curr_prompt with prompt basis
            curr_prompt = prompt_basis + sp_t + split_string

    # add last prompt to list
    all_prompts.append(curr_prompt)

    return all_prompts


# # #

def transcribe_audio(audio_file, OA_key):
    """
	transcribes audio file into text

	Input:
	audio_file: opened audiofile of any of the following formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
	OA_key - str: OpenAI key for API call

	Output:
	transcribe_out - {str:str}: dictionary with "Text" as key and the transcribed text from the audio file as value

	Notes:
	Output can be used as input for text based GPT implementations
	"""

    # set openai key
    openai.api_key = OA_key

    # Run the audio transcription function
    transcribe_out = audio_transcription_with_backoff(audio_file)

    return transcribe_out
