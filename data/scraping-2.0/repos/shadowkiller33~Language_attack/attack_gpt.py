import openai
import os
import argparse
import re
from tqdm import tqdm
import json
import aiolimiter
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
from typing import Any, Dict, List
import logging
import asyncio

from datasets import load_dataset

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
openai.api_key = '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'
# if "OPENAI_API_KEY" not in os.environ or "OPENAI_ORGAINZATION" not in os.environ:
#     raise ValueError(
#         "OPENAI_API_KEY environment variable must be set when using OpenAI API."
#     )
#openai.organization = os.getenv("OPENAI_ORGAINZATION")


MODEL = "gpt-3.5-turbo-0613"
TEMPERATURE = 0.7
MAX_TOKENS = 500


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    prompt_id: int,
    prompt: Dict[str, Any],
    save_dir: str,
) -> Dict[str, Any]:
    async with limiter:
        for _ in range(5):
            try:
                response = await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                output = prompt
                output["generated"] = response["choices"][0]["message"]["content"]
                #print(output["generated"])
                with open(f"{save_dir}/response_{prompt_id}.json", "w") as f:
                    json.dump(response, f, indent=4)
                with open(f"{save_dir}/output_{prompt_id}.json", "w") as f:
                    json.dump(output, f, indent=4)
                return (prompt_id, response)
            except openai.error.RateLimitError:
                logger.info("OpenAI API rate limit exceeded. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logger.info("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                logger.warning(f"OpenAI API error: {e}")
                break
            except openai.error.InvalidRequestError as e:
                logger.warning(f"OpenAI API error: {e}")
                break
            except openai.error.APIConnectionError as e:
                logger.warning(f"OpenAI API error: {e}")
                await asyncio.sleep(10)

        failure_msg = {"choices": [{"message": {"content": "REQUEST_FAILURE"}}]}
        return (prompt_id, failure_msg)


async def generate_from_openai_chat_completion(
    model: str,
    prompts: List[Dict],
    temperature: float,
    max_tokens: int,
    top_p: float = 1,
    requests_per_minute: int = 300,
    save_dir: str = None,
):
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = []
    bucket_size = 2000
    for prompt_id, prompt in enumerate(prompts):
        if os.path.exists(f"{save_dir}/response_{prompt_id}.json"):
            logger.info(f"Sample {prompt_id} already computed, skip")
            continue
        async_responses.append(
            _throttled_openai_chat_completion_acreate(
                model=model,
                messages=prompt["message"],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                limiter=limiter,
                prompt_id=prompt_id,
                prompt=prompt,
                save_dir=save_dir,
            )
        )
        if len(async_responses) == bucket_size:
            await tqdm_asyncio.gather(*async_responses)
            async_responses = []

    if len(async_responses) == 0:
        await session.close()
        return []

    await tqdm_asyncio.gather(*async_responses)
    await session.close()


async def handle_task(prompts, save_dir):
    output_dir = f"{save_dir}"
    os.makedirs(output_dir, exist_ok=True)
    await generate_from_openai_chat_completion(
        model=MODEL,
        prompts=prompts,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        save_dir=output_dir,
    )


def main(prompt, output_dir):
    """ prompt: a list of dictionary where each dict contains the key`message` """
    asyncio.run(handle_task(prompt, output_dir))


# TODO: write your script to construct the prompt for your dataset
import pickle
def retrieve_prompt():
    # read input data
    lang_dict = {
        'hau_Latn': 'Hausa',
        'hye_Armn': 'Armenian',
        'ibo_Latn': 'Igbo',
        'jav_Latn': 'Javanese',
        'kam_Latn': 'Kamba',
        'khk_Cyrl': 'Halh Mongolian',
        'luo_Latn': 'Luo',
        'mri_Latn': 'Maori',
        'urd_Arab': 'Urdu',
    }



    mid_lang_dict = {
        'xho_Latn': 'Xhosa',
        'sot_Latn': 'Southern Sotho',
        'slv_Latn': 'Slovenian',
        'slk_Latn': 'Slovak',
        'mlt_Latn': 'Maltese',
        'lit_Latn': 'Lithuanian',
        'est_Latn': 'Estonian',
        'bos_Latn': 'Bosnian',
        'afr_Latn': 'Afrikaans',
    }

    prompts = []
    for lang in mid_lang_dict.keys():
        with open(f'./medium/{lang}.pkl', 'rb') as f:
            data = pickle.load(f)
        language = mid_lang_dict[lang]
        system_msg = "You're an expert in the providing instructions, and I require your guidance on a few matters. " \
                     f"Could you please provide instructions to help me accomplish the following tasks? I will use {language} to describe the task, do not try to translate them, " \
                     f"and try to answer my questions in {language}.\n"
        for x in data:
            user_msg = x
            #print(user_msg)
            dict = {"message":[{'role': "user",'content': system_msg }, {'role': "system",'content': user_msg}]}
            prompts.append(dict)
    # with open('./low/hau_Latn.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # language = dict['hau_Latn']
    # system_msg = "You're an expert in the providing instructions, and I require your guidance on a few matters. " \
    #              f"Could you please provide instructions to help me accomplish the following tasks? I will use {language} to describe the task, do not try to translate them, " \
    #              f"and try to answer my questions in {language}.\n"
    #
    #
    # for x in data[:1]:
    #     user_msg = x
    #     print(user_msg)
    #     dict = {"message":[{'role': "user",'content': system_msg }, {'role': "system",'content': user_msg}]}
    #     prompts.append(dict)

    return prompts


if __name__ == "__main__":
    from distutils.util import strtobool

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    ### add your args

    args = parser.parse_args()
    prompt = retrieve_prompt()
    main(prompt, args.output_dir)

