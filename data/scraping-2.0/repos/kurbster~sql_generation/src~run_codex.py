#!/usr/bin/env python3
import sys
import time
import logging

from typing import List, Dict
from collections import deque
from pathlib import Path
PARENT_DIR = Path(__file__, '../..').resolve()
sys.path.append(str(PARENT_DIR))

import hydra
import openai

from omegaconf import OmegaConf
from openai.error import RateLimitError, ServiceUnavailableError

from src.config import (
    APIConfig, ExperimentConfig, CodexGenerationConfig,
    get_output_dir, get_exp_time
)
from lib.write_output import OutputManager
from lib.create_few_shot_prompt import generate_codex_prompts

logger = logging.getLogger("myLogger")

def generate_sql(api_cfg: APIConfig, gen_cfg: CodexGenerationConfig, prompts: List[Dict[str, str]]):
    data_output_dir = PARENT_DIR / gen_cfg.data_output_dir
    api_cfg = OmegaConf.to_container(api_cfg)

    # Create output manager object to write output to disk
    output_dir = get_output_dir()
    gpt_response_dir = output_dir / "gpt_input_output"

    exp_time = get_exp_time()
    output_queues = {}
    output_managers = {}
    # TODO: Find a better way to do this. We need to store the output
    # of codex in a subdir that represents the input dataset.
    # i.e generated, spider, wikiSQL
    for dname in gen_cfg.input_data_files:
        new_data_output = data_output_dir / dname
        new_data_output.mkdir(parents=True, exist_ok=True)
        new_response_output = gpt_response_dir / dname
        new_response_output.mkdir(parents=True)
        output_managers[dname] = OutputManager(
            exp_time=exp_time,
            exp_output_dir=new_response_output,
            data_output_dir=new_data_output
        )
        output_queues[dname] = deque()

    def call_model(prompt: str, db_name: str, dataset_name: str, itr: int):
        response = openai.Completion.create(
            prompt=prompt['text'],
            **api_cfg
        )
        output_queues[dataset_name].append({(db_name, 'response', itr): response})
        results = parse_response(response)
        for i, result in enumerate(results):
            result['input'] = prompt['text']
            # Every iteration might have multiple generations.
            # So we save them under a different iteration name
            output_queues[dataset_name].append({(db_name, 'input_output', f'{itr}_{i}'): result})
            # Save the question query pair directly to the data dir
            new_prompt = prompt.copy()
            new_prompt['gen_sql'] = result['gen_sql']
            new_prompt['n_generation'] = i
            # output_queues[dataset_name].append({(db_name, 'pair', f'{itr}_{i}'): prompt})
            output_queues[dataset_name].append({(db_name, 'pair', f'{itr}_{i}'): new_prompt})

    try:
        for itr, prompt in enumerate(prompts):
            db_name = prompt['db_id']
            dataset_name = prompt["dataset_name"]
            logger.info(f'Generating for dataset: {dataset_name} and db: {db_name}')
            try:
                call_model(prompt, db_name, dataset_name, itr)
            except RateLimitError as e:
                if str(e) == "You exceeded your current quota, please check your plan and billing details.":
                    logger.error('We ran out of tokens :(')
                    raise e
                logger.error('We have hit our rate limit. Writing output then sleeping.')
                output_queue = output_queues[dataset_name]
                output_manager = output_managers[dataset_name]
                seconds_spent = output_manager.write_output(output_queue)
                logger.debug(f'Spent: {seconds_spent} seconds writing output')
                time.sleep(90 - seconds_spent)
                call_model(prompt, db_name, dataset_name, itr)
            except ServiceUnavailableError as e:
                logger.error(e)
                logger.error('The service is unavailable sleeping for a minute.')
                time.sleep(600)
                call_model(prompt, db_name, dataset_name, itr)

    except Exception as e:
        for dname in gen_cfg.input_data_files:
            output_managers[dataset_name].write_output(output_queues[dataset_name])
        raise e
    
    # FIXME: Related to the TODO above. This should be easier
    # We must iterate through the output queue to find the proper manager
    for dname in gen_cfg.input_data_files:
        output_managers[dataset_name].write_output(output_queues[dataset_name])

def parse_response(response: Dict[str, str]) -> List[Dict[str, str]]:
    responses = []
    for choice in response['choices']:
        output = choice['text']
        query = output.replace('\n#', '').strip()
        # This is the case that the query is wrapped inside ```
        if '```' in output:
            query = query.split('```')[1]
            query = ' '.join(query.splitlines()[1:])
        query_check = query.split(';')
        # Checking if there was something after the query
        if len(query_check) > 1 and query_check[1] != '':
            logger.warning(f"""Found data after the sql query
            Full output: {output}
            Parsed query: {query}
            new_query: {query_check[0]}
            """)
            query = query_check[0]
        responses.append({'output': output, 'gen_sql': query})
    return responses

@hydra.main(config_path="configs", config_name="codex", version_base="1.2")
def main(cfg: ExperimentConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    
    prompts = generate_codex_prompts(cfg.generation_cfg)

    logger.info(f'Generating {len(prompts)} sql queries.')
    generate_sql(cfg.api_cfg, cfg.generation_cfg, prompts)
    
if __name__ == '__main__':
    main()
