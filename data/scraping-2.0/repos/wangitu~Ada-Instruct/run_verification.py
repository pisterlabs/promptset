import sys
import os
import json
import fire
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from openai_sandbox.execution import check_correctness

from config import get_verification_config
from tasks import get_task


def verify_code(code_completions, task, n_workers=8, timeout=3.0):
    
    def check_program(problem: Dict, completion: str):
        reference = '\n'.join(problem['test_list'])
        return completion + '\n' + reference
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        results = []
        
        for code_completion in tqdm(code_completions):
            if (preparation := task.prepare_for_verification(code_completion)) is None:
                continue
            (problem, completion) = preparation
            problem['task_id'] = code_completion['original_id']
            future = executor.submit(check_correctness, problem, completion, timeout, check_program=check_program)
            futures.append(future)
            
        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)
    
    id2code_completion = {code_completion['original_id']: code_completion for code_completion in code_completions}
    verified_completions = []
    for result in results:
        if result['passed']:
            verified_completions.append(id2code_completion[result['task_id']])
            
    return verified_completions


VERIFIERS = {
    'mbpp': verify_code ,
    'humaneval': verify_code
}


def verify(
    task_name: str,
    in_file: str,
    out_file: str,
    do_task_verification: bool = False
):
    with open(in_file, 'r', encoding='utf-8') as f:
        completions = json.load(f)['completions']
    
    valid_completions = [completion for completion in completions if completion.pop('valid')]
    
    if do_task_verification and task_name in VERIFIERS:
        task = get_task(task_name)
        args, kwargs = get_verification_config(task_name)
        verified_completions = VERIFIERS[task_name](valid_completions, task, *args, **kwargs)
    else:
        verified_completions = valid_completions
    
    print(f"len(verified_completions) = {len(verified_completions)}")
    
    out_dir, _ = os.path.split(out_file)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(verified_completions, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    fire.Fire(verify)
    