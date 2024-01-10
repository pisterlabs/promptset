#!/usr/bin/env python3
import os
import json
import time
import hydra
import openai
import logging

from typing import List
from omegaconf import OmegaConf
from openai.error import RateLimitError, InvalidRequestError

from lib.config import ExperimentConfig, register_configs
from lib.prompt_generation import generate_summary_prompt

logger = logging.getLogger('apiLogger')

def get_summaries(cfg: ExperimentConfig, prompts: List[str]):
    """Make a summary using GPT.

    Args:
        config (Path): Path to the config file.

    Raises:
        Exception: If your prompt is too long.
    """
    api_cfg = OmegaConf.to_container(cfg.api_params)

    API_KEY = os.getenv("OPENAI_API_KEY")
    logger.info(f'using apikey: {API_KEY}')
    openai.api_key = API_KEY

    try:
        response = openai.Completion.create(
            prompt=prompts,
            **api_cfg
        )
    except RateLimitError as e:
        logger.debug(e)
        if str(e) == 'you exceeded your current quota, please check your plan and billing details':
            logger.error('We ran out of tokens')
            return
        logger.error('We have hit our rate limit. Sleeping for a minute...')
        time.sleep(60)
        response = get_summaries(cfg, prompts)

    return response
    
@hydra.main(config_path="configs", config_name="gpt3")
def main(cfg):
    prompts, extras, output_dirs = generate_summary_prompt(cfg.generation_params)

    summaries = []
    try:
        for i in range(0, len(prompts), 20):
            logger.info(f'Generating summaries for problems {i} through {i+20}')
            response = get_summaries(cfg, prompts[i: i+20])
            if response is None:
                break
            summaries += [val["text"] for val in response["choices"]]
    except InvalidRequestError as e:
        logger.critical('Send an invalid request')
        for p, out in zip(prompts, output_dirs):
            logger.debug(f'Prompt for problem {out}')
            logger.debug(f'\n{p}\n')
        logger.critical(e)
    finally:
        for prompt, result, extra, output_dir in zip(prompts, summaries, extras, output_dirs):
            logger.info(f'Writing result to {output_dir}')
            with open(f'{output_dir}/output.json', 'w') as f:
                json.dump(response, f, indent=4)
            with open(f'{output_dir}/{cfg.generation_params.summary_types[0]}.txt', 'w') as f:
                f.write(result+'\n'+extra)
            with open(f'{output_dir}/input_to_gpt.txt', 'w') as f:
                f.write(prompt)
    
if __name__ == '__main__':
    register_configs()
    main()