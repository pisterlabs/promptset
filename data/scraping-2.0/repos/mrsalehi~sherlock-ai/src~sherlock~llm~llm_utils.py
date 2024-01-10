import os
import time
import sys
import openai
from sherlock.llm.prompts import SYSTEM_PROMPT, USER_PROMPT

from multiprocessing import Pool
from tqdm import tqdm
import logging

from openai.error import RateLimitError, ServiceUnavailableError
import backoff
from typing import Dict, List, Tuple, Union


# openai.api_key = os.environ.get")
openai.api_key = "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"  # expired key ;)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# class MultiProcessCaller:
#     @staticmethod
#     def call_multi_process(fn, data, num_processes=8):
#         result = []
#         p = Pool(num_processes)
#         pbar = tqdm(total=len(data))
#         for i in p.imap_unordered(fn, data):
#             if i is not None:
#                 result.append(i)
#             pbar.update()

#         return result

# def read_img(filepath):
#     if os.path.isfile(filepath):
#         raw_image = Image.open(filepath)
#     else:
#         raw_image = Image.open(requests.get(filepath, stream=True).raw)
#     raw_image = raw_image.convert("RGB")
    
#     return raw_image


# def read_arrow(path):
#     table = pa.ipc.RecordBatchFileReader(
#         pa.memory_map(path, "r")
#     ).read_all()
#     return table
    
# def highlight_region(image, bboxes):
#     image = image.convert('RGBA')
#     overlay = Image.new('RGBA', image.size, '#00000000')
#     draw = ImageDraw.Draw(overlay, 'RGBA')
#     for bbox in bboxes:
#         print(bbox)
#         if isinstance(bbox, dict):
#             x = bbox['left']
#             y = bbox['top']
#             draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
#                             fill='#ff05cd3c', outline='#05ff37ff', width=3)
#         else:
#             draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])],
#                             fill='#ff05cd3c', outline='#05ff37ff', width=3)
#     image = Image.alpha_composite(image, overlay)
#     return image.convert('RGB')

# class OpenaiAPI:

#     def __init__(self, api_key) -> None:
#         openai.api_key = api_key

    
@backoff.on_exception(backoff.expo, (RateLimitError, ServiceUnavailableError), max_time=60)
def complete_chat(messages, model='gpt-3.5-turbo', max_tokens=256,  num_log_probs=None,  n=1, 
                top_p = 1.0, temperature=0.5, stop = None, echo=True,
                frequency_penalty = 0., presence_penalty=0. ):

    # call GPT-3 API until result is provided and then return it
    response = None
    c = 0
    while c < 100:
        try:
            response = openai.ChatCompletion.create(messages=messages, model=model, max_tokens=max_tokens, temperature=temperature,
                                                    stop=stop, n=n, top_p=float(top_p),
                                                frequency_penalty = frequency_penalty,
                                                presence_penalty= presence_penalty)
            return response
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                print(f"InvalidRequestError\nQuery:\n\n{messages}\n\n")
                print(sys.exec_info())
                break
            else:
                print('Error:', sys.exc_info())
                time.sleep(5)
                c+=1

    return response

    
def generate_response(
        query, 
        # documents_md_doc, 
        # documents_discord, 
        # documents_md_nb,
        context_docs: Dict={},
        model='gpt-3.5-turbo',
    ):
    """
    gets the query as a string and a set of documents as a list of strings and generates the answer
    based on the documents

    
    There are different ways to get the response from the LLM based on documents:
    1. Pass all the top documents from each source of data as the context (all_in_one_context)
    2. Generate the response for each source of data separately and then combine the responses 
       again as the context and generate the final response (separate_context)
    """
    # doc_string_documentation = ""
    # for doc in documents_md_doc:
    #     doc_string_documentation += f"MARKDOWN DOCUMENTS: {doc}\n\n"
    
    # doc_string_discord = ""
    # for doc in documents_discord:
    #     doc_string_discord += f"DISCORD MESSAGES: {doc}\n\n"

    # doc_string_nb = ""
    # for doc in documents_md_nb:
    #     doc_string_nb += f"JUYPTER NOTEBOOKS: {doc}\n\n"

    llm_context = "" 
    for context_k, context_v in context_docs.items():
        _context_string = f"{context_k}:\n"
        for doc in context_v:
            _context_string += f"{doc}\n\n"
        # logger.info(f"context for {context_k}: {_context_string}")
        llm_context += _context_string

    # prompt = SYSTEM_PROMPT + USER_PROMPT.format(query, doc_string_documentation, doc_string_discord)
    prompt = USER_PROMPT.format(query, llm_context)
    
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": prompt
    }]

    response = complete_chat(messages, model=model, max_tokens=2047)

    logger.info("Here is Sherlock's response:")
    logger.info(response.choices[0]["message"]["content"])