import pdb

import json
import time
import sys
from copy import deepcopy

import openai
from openai import AsyncOpenAI, OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import create_base_retry_decorator
from langchain.schema import Document
from typing import List, Tuple, Any, Callable

import tiktoken
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_model_max_len(model_name:str) -> int:
    with open("model_config.json", "r") as f:
        config = json.load(f)

    model_max_len = config[model_name]["max_allowed_token_len"]
    tokens_for_prompt = config["tokens_for_prompt_and_generation"]

    return model_max_len - tokens_for_prompt


def get_model_max_tokens(model_name:str) -> int:
    with open("model_config.json", "r") as f:
        config = json.load(f)

    model_max_len = config[model_name]["max_allowed_token_len"]

    return model_max_len


def get_time_encoded_transcripts(transcript: List[dict],
                                 model_name: str) -> Tuple[bool, float, float, str]:
    """
    Create subsets of transcripts based on the maximum token length of the model used.
    If the model maximum token length is 5000 and total tokens in the transcript are
    10,000. Then create 2 transcripts with each 5000 tokens with their respective
    start and end timings.

    :param transcript: transcript of a video
    :param model_name: the model name selected by the user
    :return: a tuple of (bool value highlighting where the transcript was split or not,
            start timing of the split of the transcript,
            end timing of the split of the transcript,
            the transcript split)
    """

    enc = tiktoken.encoding_for_model(model_name)
    model_max_token_len = get_model_max_len(model_name)

    sentences = []
    length_till_now = 0
    all_start = 0.0
    was_transcript_splitted = False

    for dialogue in transcript:
        text = dialogue["text"]
        start = dialogue["start"]
        duration = dialogue["duration"]

        # To include for tokens added by joining all sentences.
        enc_text = enc.encode(text + "\n")

        if len(enc_text) + length_till_now > model_max_token_len:
            was_transcript_splitted = True
            yield was_transcript_splitted, all_start, start + duration, "\n".join(sentences)
            sentences = [text]
            all_start = start + duration
            length_till_now = len(enc_text)

        else:
            sentences.append(text)
            length_till_now += len(enc_text)

    yield was_transcript_splitted, all_start, start + duration, "\n".join(sentences)


def get_documents(video_ids: List[str],
                  video_titles: List[str],
                  transcripts: List[List[dict]],
                  model_name: str) -> List[Document]:
    """
    Return a list of documents from transcripts with their video id, video title as metadata of the document

    :param video_ids: List of video ids of the video
    :param video_titles: List of names of the videos
    :param transcripts: List of transcript of each video
    :param model_name: model name selected by user for summarization
    :return: List of documents created with transcript text and relevant metadata
    """


    documents = []

    for video_id, video_title, transcript in zip(video_ids, video_titles, transcripts):
        for did_split_happen, start, end, text in get_time_encoded_transcripts(transcript, model_name):
            start_min, start_sec = int(start / 60), int(start % 60)
            end_min, end_sec = int(end / 60), int(end % 60)
            video_start = abs(int(start))

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "https://www.youtube.com/watch?v=" + video_id,
                        "video_start": video_start,
                        "start_min": start_min,
                        "start_sec": start_sec,
                        "end_min": end_min,
                        "end_sec": end_sec,
                        "did_split_happen": did_split_happen,
                        "title": video_title
                    }
                )
            )

    return documents


def divide_big_summary_into_parts(summary: str, model_name: str) -> List[str]:
    """
    Function to divide a big summary into smaller parts based on the max token length supported
    by the model.
    While processing a lot of videos, the token length of the combination of all smaller summaries could
    be more than the max token length supported by the model. So, we divide the combination into smaller
    parts. This allows model to process the summaries

    :param summary: combined summary created by joining multiple smaller sumamries
    :param model_name: model name used for summarizartion
    :return: list of smaller summaries
    """
    enc = tiktoken.encoding_for_model(model_name)
    model_max_token_len = get_model_max_len(model_name)

    encoded_summary = enc.encode(summary)

    smaller_summaries = []
    if len(encoded_summary) > model_max_token_len:
        for i in range(0, len(encoded_summary), model_max_token_len):
            smaller_summaries.append(
                enc.decode(encoded_summary[i:i + model_max_token_len])
            )
    else:
        smaller_summaries = [summary]

    return smaller_summaries


def get_updated_prompts(original_prompt_dict: dict,
                        context: str,
                        summary_keywords: Any = None) -> dict:
    """
    Format the original prompt string with context and summary keywords.
    The original prompts are normal string with placeholders for context and summary_keywords.
    Format the string with actual values.

    :param original_prompt_dict: prompt dict for the use case
    :param context: the transcript or the combination of summaries from transcripts of the videos
    :param summary_keywords: keywords to focus on while summarization if present
    :return:
    """
    prompt_dict = deepcopy(original_prompt_dict)

    if prompt_dict["summary_keywords"]:
        assert summary_keywords is not None
        prompt_dict["system"] = prompt_dict["system"].format(summary_keywords=summary_keywords)

    prompt_dict["user"] = prompt_dict["user"].format(context=context)

    return prompt_dict


def get_max_tokens(text: str, model_name:str) -> int:
    """
    Return exact number of maximum tokens that the model can use for generation
    total size of input prompt and the size of transcripts can vary.

    We subtract and additional 20 token from the possible limits due to
    some mismatch in values we get from tiktoken and actual usage from Openai API.

    :param text: text to be passed to the model
    :param model_name: model name for summarization
    :return: possible maximum tokens for generation
    """

    model_max_tokens = get_model_max_tokens(model_name)

    enc = tiktoken.encoding_for_model(model_name)
    enc_text = enc.encode(text)

    # Setting 1500 to allow for more output from GPT 4 turbo models with 128k context len
    return min(model_max_tokens - len(enc_text) - 20, 1500)


def _create_retry_decorator(
    max_tries: int,
    run_manager: Any = None,
) -> Callable[[Any], Any]:
    """
    Retry decorator from langchain lib that handles possible errors from OpenAI API and allows retries.

    :param max_tries: max retries allowed at failure time
    :param run_manager: run manager which contains callbacks that are passed to LLMs for generation. Not applicable here.
    :return: a decorator that allows retries.
    """
    import openai

    errors = [
        openai.Timeout,
        openai.APIError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.APIStatusError,
        openai.InternalServerError
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=max_tries, run_manager=run_manager
    )


async def acompletion_with_retry(max_tries=6,
                                 run_manager=None,
                                 **kwargs: Any) -> str:
    """
    Function to apply retry decorator on async calls to OpenAI

    :param max_tries: max retries allowed at failure time
    :param run_manager: run manager which contains callbacks that are passed to LLMs for generation. Not applicable here.
    :param kwargs: variables to be passed to async call function
    :return: async call function with retry decorator.
    """
    retry_decorator = _create_retry_decorator(max_tries, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await aget_response_from_llm(**kwargs)

    return await _completion_with_retry(**kwargs)


async def aget_response_from_llm(model_name: str,
                                 prompt_dict: dict,
                                 max_tokens: int,
                                 stream: bool = True
                                 ) -> str:
    """
    Async call function for accessing OpenAI completion endpoints.

    :param model_name: model to be used
    :param prompt_dict: the prompt to be used for 'system' and 'user' roles
    :param max_tokens: the maximum number of output tokens in model response
    :param stream: whether to stream the generated answer or not
    :return: returns the completion answer string from OpenAI
    """

    client = AsyncOpenAI()

    # get async generator for chat completion
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt_dict["system"]},
            {'role': 'user', 'content': prompt_dict["user"]}
        ],
        temperature=0,
        max_tokens=max_tokens,
        stream=stream
    )

    if not stream:
        # If stream is set to False, then the answer is returned in a single chunk.
        output = response.choices[0].message.content
    else:
        # If stream is set to true then one has to iterate over each token individually.
        # We have to flush the token stdout, that allows the tokens to be present in the UI while
        # iterating over each token returned by OpenAI.

        collected_response = []
        async for chunk in response:
            try:
                finish = chunk.choices[0].finish_reason
                if finish == "stop" or finish == "length":
                    break
                chunk_message = chunk.choices[0].delta.content  # extract the message
                sys.stdout.write(chunk_message)
                sys.stdout.flush()
                collected_response.append(chunk_message)
            except Exception as e:
                #pdb.set_trace()
                print(e)
                continue

        output = "".join(collected_response)

    return output


def get_summary_with_keywords(documents: List[Document],
                              keywords: List[str],
                              per_document_template: PromptTemplate,
                              combine_document_template: PromptTemplate,
                              open_ai_model: str,
                              total_videos: int) -> str:
    """
    Method to return summary of documents with focus on specific keywords.
    This method uses langchain to query LLM.

    :param documents: List of documents containing transcripts of videos along with video metadata
    :param keywords: keywords to focus on for summarization
    :param per_document_template: Prompt template to be used for each video
    :param combine_document_template: Prompt template to be used for combined summary of videos.
    :param open_ai_model: model to use for summarization.
    :return: the output from llm
    """

    # to stream answer we use StreamingStdOutCallbackHandler which prints each new token to stdout
    llm = ChatOpenAI(model_name=open_ai_model,
                     temperature=0.0,
                     streaming=True,
                     callbacks=[StreamingStdOutCallbackHandler()]
                     )

    summary_keywords = ", ".join(keywords)

    per_document_llm_chain = LLMChain(llm=llm, prompt=per_document_template)

    smaller_summaries = []
    for i, d in enumerate(documents):
        logger.info(f'Summary {i}:\n')
        print('\n')

        if d.metadata["did_split_happen"]:

            # If transcript was too long for the model to process in one time,
            # process subset of the transcript and show the start and end time for the subset

            print(f'Summary of video [{d.metadata["title"]}]({d.metadata["source"]}&t={d.metadata["video_start"]}s)'
                  f' from {d.metadata["start_min"]}:{d.metadata["start_sec"]} to '
                  f'{d.metadata["end_min"]}:{d.metadata["end_sec"]} \n')
            source_doc = d.metadata["source"] + f"&t={d.metadata['video_start']}s"
        else:
            print(f'Summary of video [{d.metadata["title"]}]({d.metadata["source"]})\n')
            source_doc = d.metadata["source"]

        d_summary = per_document_llm_chain.run(context=d.page_content, summary_keywords=summary_keywords)
        smaller_summaries.append((source_doc, d_summary))

    big_summary = ""
    for source, summary in smaller_summaries:
        big_summary += f"Source: {source}\n Summary: {summary}\n\n"

    if total_videos == 1:
        # If there is only a single video then we don't need to create a combined summary
        return big_summary

    logger.info("Generating consolidated summary based on your search topics")
    print('\n')
    print("##### Generating consolidated summary based on your search topics")
    print('\n')

    llm_2 = ChatOpenAI(model_name=open_ai_model,
                       temperature=0.0,
                       streaming=True,
                       )

    combined_document_chain = LLMChain(llm=llm_2, prompt=combine_document_template)

    summaries = divide_big_summary_into_parts(big_summary, open_ai_model)

    while len(summaries) > 1:
        # Process a big summary in sub parts till the combined summary is small enough
        smaller_chunks = ""
        for smaller_summary in summaries:
            smaller_chunks += combined_document_chain.run(doc_summaries=smaller_summary,
                                                          summary_keywords=summary_keywords)

        summaries = divide_big_summary_into_parts(smaller_chunks, open_ai_model)

    return combined_document_chain.run(doc_summaries=summaries[0],
                                       summary_keywords=summary_keywords,
                                       callbacks=[StreamingStdOutCallbackHandler()])


async def aget_summary_with_keywords(documents: List[Document],
                                     keywords: List[str],
                                     per_document_template: dict,
                                     combine_document_template: dict,
                                     open_ai_model: str,
                                     total_videos: int) -> str:

    """
    Async method to generate summary around fixed keywords.
    This method doesn't use langchain but communicates with OpenAI directly

    :param documents: List of documents containing transcripts of videos along with video metadata
    :param keywords: keywords to focus while summarization
    :param per_document_template: prompt dict with system and user prompts to be used while processing a single video
    :param combine_document_template: prompt dict with system and user prompts to be used while processing combined summary
            from different videos
    :param open_ai_model: model to be used for summarization
    :param total_videos: total videos being processed
    :return: the result summary
    """

    summary_keywords = ", ".join(keywords)

    smaller_summaries = []
    for i, d in enumerate(documents):
        logger.info(f'Summary {i}:\n')
        print('\n')

        if d.metadata["did_split_happen"]:
            print(f'Summary of video [{d.metadata["title"]}]({d.metadata["source"]}&t={d.metadata["video_start"]}s)'
                  f' from {d.metadata["start_min"]}:{d.metadata["start_sec"]} to '
                  f'{d.metadata["end_min"]}:{d.metadata["end_sec"]} \n')
            source_doc = d.metadata["source"] + f"&t={d.metadata['video_start']}s"
        else:
            print(f'Summary of video [{d.metadata["title"]}]({d.metadata["source"]})\n')
            source_doc = d.metadata["source"]

        prompt_dict = get_updated_prompts(per_document_template, d.page_content, summary_keywords)
        max_tokens = get_max_tokens(prompt_dict["system"] + prompt_dict["user"], open_ai_model)

        d_summary = await acompletion_with_retry(model_name=open_ai_model,
                                                 prompt_dict=prompt_dict,
                                                 max_tokens=max_tokens)

        smaller_summaries.append((source_doc, d_summary))

    big_summary = ""
    for source, summary in smaller_summaries:
        big_summary += f"Source: {source}\n Summary: {summary}\n\n"

    if total_videos == 1:
        # If there is only a single video then we don't need to create a combined summary
        return big_summary

    logger.info("Generating consolidated summary based on your search topics")
    print('\n')
    print("##### Generating consolidated summary based on your search topics")
    print('\n')

    summaries = divide_big_summary_into_parts(big_summary, open_ai_model)

    while len(summaries) > 1:
        smaller_chunks = ""
        for smaller_summary in summaries:

            prompt_dict = get_updated_prompts(combine_document_template, smaller_summary, summary_keywords)
            max_tokens = get_max_tokens(prompt_dict["system"] + prompt_dict["user"], open_ai_model)

            smaller_chunks += await acompletion_with_retry(model_name=open_ai_model,
                                                           prompt_dict=prompt_dict,
                                                           max_tokens=max_tokens,
                                                           stream=False)

        summaries = divide_big_summary_into_parts(smaller_chunks, open_ai_model)

    prompt_dict = get_updated_prompts(combine_document_template, summaries[0], summary_keywords)
    max_tokens = get_max_tokens(prompt_dict["system"] + prompt_dict["user"], open_ai_model)

    final_summary = await acompletion_with_retry(model_name=open_ai_model,
                                                 prompt_dict=combine_document_template,
                                                 max_tokens=max_tokens)
    return final_summary


def get_summary_of_each_video(documents: List[Document],
                              per_document_template: PromptTemplate,
                              open_ai_model: str) -> str:
    """
    Return a summary of each video separately. This function returns a general summary
    and does not create a summary around specific keywords.
    This function uses langchain backend.

    :param documents: List of documents containing transcripts of videos along with video metadata
    :param per_document_template: Prompt template to be used for each video
    :param open_ai_model: model to be used for summarization
    :return: result summary
    """

    llm = ChatOpenAI(model_name=open_ai_model,
                     temperature=0.0,
                     streaming=True,
                     callbacks=[StreamingStdOutCallbackHandler()]
                     )

    per_document_llm_chain = LLMChain(llm=llm, prompt=per_document_template)

    summary = ""
    for i, d in enumerate(documents):
        logger.info(f'Summary {i}:\n')
        print('\n')
        if d.metadata["did_split_happen"]:
            print(f'Summary of video [{d.metadata["title"]}]({d.metadata["source"]}&t={d.metadata["video_start"]}s)'
                  f' from {d.metadata["start_min"]}:{d.metadata["start_sec"]} to '
                  f'{d.metadata["end_min"]}:{d.metadata["end_sec"]} \n')
        else:
            print(f'Summary of video [{d.metadata["title"]}]({d.metadata["source"]})\n')

        d_summary = per_document_llm_chain.run(context=d.page_content)
        summary += d_summary
        summary += f"\n\nSource: {d.metadata['source']}"
        if d.metadata["did_split_happen"]:
            summary += f"&t={d.metadata['video_start']}s"
        summary += "\n"

    return summary


async def aget_summary_of_each_video(documents: List[Document],
                                     per_document_template: dict,
                                     open_ai_model: str) -> str:
    """
    Async method to return a summary of each video separately. This function returns a
    general summary and does not create a summary around specific keywords.
    This function directly hits OpenAI API and doesnt use langcahin backend.

    :param documents: List of documents containing transcripts of videos along with video metadata
    :param per_document_template: dict with "system" and "user" prompts to generate summary for each video
    :param open_ai_model: model to use for summarization
    :return: the result summary
    """

    summary = ""
    for i, d in enumerate(documents):
        logger.info(f'Summary {i}:\n')
        print('\n')
        if d.metadata["did_split_happen"]:
            print(f'Summary of video [{d.metadata["title"]}]({d.metadata["source"]}&t={d.metadata["video_start"]}s) from '
                  f'{d.metadata["start_min"]}:{d.metadata["start_sec"]} to '
                  f'{d.metadata["end_min"]}:{d.metadata["end_sec"]} \n')
        else:
            print(f'Summary of video [{d.metadata["title"]}]({d.metadata["source"]})\n')

        prompt_dict = get_updated_prompts(per_document_template, d.page_content)
        max_tokens = get_max_tokens(prompt_dict["system"] + prompt_dict["user"], open_ai_model)

        d_summary = await acompletion_with_retry(model_name=open_ai_model,
                                                 prompt_dict=prompt_dict,
                                                 max_tokens=max_tokens)

        summary += d_summary
        summary += f"\n\nSource: {d.metadata['source']}"
        if d.metadata["did_split_happen"]:
            summary += f"&t={d.metadata['video_start']}s"
        summary += "\n"

    return summary
