#!/usr/bin/env python3
import os
import json
import time
import hydra
import shutil
import openai
import logging

from typing import Dict, List
from pathlib import Path
from openai.error import RateLimitError
from omegaconf import OmegaConf

import lib.config as config

from lib import run_results, prompt_generation

logger = logging.getLogger('apiLogger')

def get_codes(prompts: List[str], cfg: config.APIConfig) -> Dict[str, any]:
    """Call the OpenAI API with the prompts given.

    Args:
        prompts (list[str]): The prompts to send to the API.
        codex_cfg (Path): The config file for the API.

    Returns:
        dict[str, any]: The response object.
    """
    logger.info(f'Using codex model: {cfg.engine}')
    
    API_KEY = os.getenv("OPENAI_API_KEY")
    logger.info(f'using apikey: {API_KEY}')
    openai.api_key = API_KEY

    api_cfg = OmegaConf.to_container(cfg)
    try:
        response = openai.Completion.create(
            prompt=prompts,
            **api_cfg
        )
    # We have exceeded OpenAIs rate limit of 150,000 tokens/min
    # We need to cleep for a minute then try again
    except RateLimitError as e:
        logger.debug(e)
        logger.error('We have hit our rate limit. Sleeping for a minute...')
        time.sleep(60)
        response = get_codes(prompts, cfg)
    return response

def save_json(dirname: Path, obj_to_save: any, fname: str, indent: int = 4) -> None:
    """Save a json object to a file

    Args:
        dirname (Path): The directory to save to.
        obj_to_save (any): The object to save.
        fname (str): The file name to save as.
        indent (int, optional): Indent of the json file. Defaults to 4.
    """
    dir_name = Path(dirname, fname)
    with open(dir_name, 'w') as f:
        json.dump(obj_to_save, f, indent=indent)

def clean_codes(codes: dict[str, str]) -> dict[str, str]:
    """We need to remove any call codex makes to the function
    it created. If not the testing module breaks.

    Args:
        codes (dict[str, str]): The codes produced by Codex.

    Returns:
        dict[str, str]: The cleaned codes.
    """
    result = {}
    for idx, code in codes.items():
        # if name main isn't causing issues. Only if
        # The function created is called. So only remove
        # The last function call if it is not inside if name main
        main_idx = code.find('if __name__')
        if main_idx == -1 and code.endswith('()'):
            # remove last line
            code_arr = code.split('\n')[:-1]
            code = '\n'.join(code_arr)
        result[idx] = code
    return result

def copy_codes(output_dir: Path, test_manifest: List[str], all_codes: Dict[str, str]):
    code_dir = output_dir.joinpath('code')
    script_dir = Path(__file__, '../tools').resolve()

    starter_code = "starter_code.py"
    solution_fname = "solutions.json"
    test_case_name = "input_output.json"
    test_case_script = "jq_script.sh"

    test_manifest = list(map(Path, test_manifest))
    for problem_path, code in zip(test_manifest, all_codes.values()):
        fname = problem_path.stem 
        problem = problem_path.parent

        problem_dir = str(problem).split('data/')[-1]
        out_dir = code_dir.joinpath(problem_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        code_str = "def code():"
        
        if (starter_path := problem.joinpath(starter_code)).exists():
            with open(starter_path) as f:
                code_str = f.read()
            
            with open(out_dir.joinpath(starter_code), 'w') as f:
                f.write(code_str)

        code_str += code

        # This will fail when there is starter code. But there isn't an easy
        # way to write in the provided function. This will be done by hand
        code_str += '\ncode()'

        out_path = out_dir.joinpath(fname+'_code.py')

        with open(out_path, 'w') as f:
            f.write(code_str)

        shutil.copy(
            problem.joinpath(test_case_name),
            out_dir.joinpath(test_case_name)
        )

        shutil.copy(
            script_dir.joinpath(test_case_script),
            out_dir.joinpath(test_case_script)
        )

        if (solutions_path := problem.joinpath(solution_fname)).exists():
            with open(solutions_path) as f:
                solutions = json.load(f)
            
            with open(out_dir.joinpath('solution.py'), 'w') as f:
                f.write(solutions[0])

def generate_codes(prompts: List[str], offset: int, cfg: config.APIConfig):
    response = get_codes(prompts, cfg)

    codes = {str(k + offset): v for k, v in enumerate([val["text"] for val in response["choices"]])}
    codes = clean_codes(codes)

    return codes, response

@hydra.main(config_path="configs", config_name="codex")
def main(cfg: config.ExperimentConfig):
    output_dir = config.get_output_dir()
    logger.info(cfg)
    logger.info(f'Saving output here {output_dir}')
    prompts, prompt_files = prompt_generation.generate_code_prompt(cfg.generation_params)

    generated_codes = dict()
    responses = []
    # Can send max of 20 prompts to codex at a time
    prompts_per_iter = cfg.generation_params.prompts_per_iter
    for i in range(0, len(prompts), prompts_per_iter):
        logger.info(f'Generating code for problems {i} through {i+prompts_per_iter}')
        new_codes, response = generate_codes(prompts[i:i+prompts_per_iter], i, cfg.api_params)
        generated_codes.update(new_codes)
        responses.append(response)

    save_json(output_dir, prompts, 'prompts.json')
    save_json(output_dir, responses, 'responses.json')
    save_json(output_dir, prompt_files, 'test.json')
    save_json(output_dir, generated_codes, 'all_codes.json')

    if cfg.generation_params.test_immediately:
        run_results.main(output_dir, cfg.generation_params.summary_types)
    else:
        if config.is_initial_job():
            run_results.prepare_agave(output_dir)

    copy_codes(output_dir, prompt_files, generated_codes)

if __name__ == "__main__":
    config.register_configs()
    main()
