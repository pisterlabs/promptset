import os
import sys
import json
import time

import numpy as np
import argparse
from typing import Callable, Dict, List
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import src.gsm_maf.feedback as feedback_utils
from src.gsm_maf.task_init import GSMInit, OSInit
from src.gsm_maf.task_iterate import GSMIterate, OSIterate
from src.utils import OPENAI_ENGINES, OS_ENGINES
from src.utils import FeedbackFactory, Logger, parse_feedback

def iterative_gsm(outfile: str, data:pd.DataFrame, prompt_dir: str, max_attempts: int, feedback_types: str, engine: str, temperature: float, batch_size: int = 5, gpus: str = "0,1", summarize_fb: bool = False, early_stop: bool = False, debug: bool = False):
    # initialize all the required components
    n_attempts = 0
    questions = data["input"]
    feedbacks_given = [ft.strip() for ft in feedback_types.split(",")]
    available_feedbacks= list(FeedbackFactory.registry.keys())
    for feedback in feedbacks_given:
        if feedback not in available_feedbacks:
            raise ValueError(f"Feedback {feedback} not found. Available feedbacks are {available_feedbacks}")

    log = [[] for i in range(len(questions))]

    feedbacks_refine = {}
    feedbacks = {}

    solutions = ["" for i in range(len(questions))]
    solutions_fixed = ["" for i in range(len(questions))]
    feedbacks_retry = [ [True for i in range(len(questions))] for j in range(len(feedbacks_given))]

    while n_attempts < max_attempts:

        # print(feedbacks_retry)
        iter_start = time.time()
        logger.write(f"Running iteration {n_attempts}")# generation of the first fast version
        if n_attempts == 0:
            logger.write("Generating initial solutions\n")

            # initialize the initial generation class
            init_prompt_path = os.path.join(prompt_dir, "init.txt")
            if engine in OPENAI_ENGINES:
                task_init = GSMInit(engine=engine, prompt_examples=init_prompt_path, temperature=temperature, max_tokens = 300)
            elif engine in OS_ENGINES:
                task_init = OSInit(engine=engine, prompt_examples=init_prompt_path, temperature=temperature, max_tokens = 300, cuda_visible_devices=gpus)

            init_gen_start = time.time()
            solutions_temp = task_init(solutions=questions, batch_size=batch_size, concurrent=True)

            usage = 0
            if type(solutions_temp) == tuple:
                usage = solutions_temp[0]
                solutions_temp = solutions_temp[1]

            for i in range(len(questions)):
                solutions[i] = solutions_temp[i]
            # print(len(solutions))
            # print(type(solutions))
            # print(type(solutions[0]))
            init_gen_end = time.time()

            mins = (init_gen_end - init_gen_start)/60
            logger.write(f"Initial generation took {mins} minutes\n")
            logger.write(f"Token usage per minute: {usage/mins}")

            # delete the task_init object
            if engine in OS_ENGINES:
                task_init.model = task_init.model.cpu()
                del task_init.model
                torch.cuda.empty_cache()
                # print(f"GPU Memory 0: {torch.cuda.memory_allocated(0)/1e9} GB")
                # print(f"GPU Memory 1: {torch.cuda.memory_allocated(1)/1e9} GB")
                # print(f"GPU Memory 2: {torch.cuda.memory_allocated(2)/1e9} GB")

            del task_init

        solutions_fixed = [solution for solution in solutions]
        for i, feedback in enumerate(feedbacks_given):
            # print(fm.prompt)
            if feedback in ["syntax"]:
                fm = FeedbackFactory.create_feedback(feedback)
            elif "os" not in feedback:
                fb_prompt_path = os.path.join(prompt_dir, f"{feedback}.txt")
                fm = FeedbackFactory.create_feedback(feedback, prompt_examples=fb_prompt_path, engine=engine, temperature=temperature)
            else:
                feedback_file = feedback.removesuffix("_os")
                fb_prompt_path = os.path.join(prompt_dir, f"{feedback_file}.txt")
                fm = FeedbackFactory.create_feedback(feedback, prompt_examples=fb_prompt_path, engine=engine, temperature=temperature, cuda_visible_devices=gpus)

            fb_gen_start = time.time()
            if any(feedbacks_retry[i]):
                # initialize the feedback modules
                if fm.eager_refine:
                    feedbacks_refine[fm.name] = ["" for i in range(len(questions))]
                else:
                    feedbacks[fm.name] = ["" for i in range(len(questions))]
                
                actual_batch_size = batch_size
                if fm.type == "lm":
                    if fm.max_tokens > 300:
                        actual_batch_size = 10

                    logger.write(f"Generating {fm.name}\n")
                    logger.write(f"Args for feedback - temperature: {fm.temperature}, max_tokens: {fm.max_tokens}, engine: {fm.engine}, batch_size: {batch_size}\n")

                # call the feedback module
                retry_idxs = list(np.where(feedbacks_retry[i])[0])
                logger.write(f"Stopped {fm.name} generation for {len(questions) - len(retry_idxs)} questions\n")
                solutions_retry = [solutions_fixed[idx] for idx in retry_idxs]
                fb_and_maybe_solns = fm(solutions=solutions_retry, batch_size=actual_batch_size, concurrent=True)

                usage = 0
                if type(fb_and_maybe_solns) == tuple:
                    usage = fb_and_maybe_solns[0]
                    fb_and_maybe_solns = fb_and_maybe_solns[1]

                # if eager_refine is on, get the solutions and feedbacks
                for j, idx in enumerate(retry_idxs):
                    # print(j, idx)
                    if early_stop and "it is correct" in fb_and_maybe_solns[j]['feedback'].lower():
                        feedbacks_retry[i][idx] = False
                    if fm.eager_refine:
                        solutions_fixed[idx] = fb_and_maybe_solns[j]["solution"]

                        if summarize_fb:
                            feedbacks_refine[fm.name][idx] = parse_feedback(fb_and_maybe_solns[j]["feedback"])
                        else:
                            feedbacks_refine[fm.name][idx] = fb_and_maybe_solns[j]["feedback"]
                    else:
                        if summarize_fb:
                            feedbacks[fm.name][idx] = parse_feedback(fb_and_maybe_solns[j]['feedback'])
                        else:
                            feedbacks[fm.name][idx] = fb_and_maybe_solns[j]['feedback']
            fb_gen_end = time.time()

            # delete the feedback module
            mins = (fb_gen_end - fb_gen_start)/60
            logger.write(f"{fm.name} generation took {mins} minutes\n")
            logger.write(f"Token usage per minute: {usage/mins}")

            if fm.type == "lm" and engine in OS_ENGINES:
                fm.model = fm.model.cpu()
                del fm.model
                del fm.tokenizer
                torch.cuda.empty_cache()

            del fm

        # only call iterate if there is at least one feedback without eager_refine
        if len(feedbacks):
            logger.write("Generating refined solutions\n")

            # initialize the refinement class
            if summarize_fb:
                iterate_prompt_path = os.path.join(prompt_dir, "iterate_summarized_feedback.txt")
            else:
                iterate_prompt_path = os.path.join(prompt_dir, "iterate.txt")
            if engine in OPENAI_ENGINES:
                task_iterate = GSMIterate(engine=engine, prompt_examples=iterate_prompt_path, temperature=temperature, max_tokens = 300)
            elif engine in OS_ENGINES:
                task_iterate = OSIterate(engine=engine, prompt_examples=iterate_prompt_path, temperature=temperature, max_tokens = 300, cuda_visible_devices=gpus)

            refine_gen_start = time.time()
            solutions_fixed_temp = task_iterate(solutions=solutions_fixed, feedbacks=feedbacks, batch_size=batch_size, concurrent=True)

            usage = 0
            if type(solutions_fixed_temp) == tuple:
                usage = solutions_fixed_temp[0]
                solutions_fixed_temp = solutions_fixed_temp[1]

            for i in range(len(questions)):
                solutions_fixed[i] = solutions_fixed_temp[i]
                if solutions_fixed[i] == "":
                    print(f"No fixed solution for example {i}")

            refine_gen_end = time.time()

            mins = (refine_gen_end - refine_gen_start)/60
            logger.write(f"Refined generation took {mins} minutes\n")
            logger.write(f"Token usage per minute: {usage/mins}")
            if engine in OS_ENGINES:
                task_iterate.model = task_iterate.model.cpu()
                del task_iterate.model
                torch.cuda.empty_cache()

            del task_iterate

        for i in range(len(questions)):
            solution = solutions[i]
            solution_fixed = solutions_fixed[i]
            feedback_with_refinement = {}
            feedback = {}
            for ft, fb in feedbacks.items():
                feedback[ft] = fb[i]
            for ft, fb in feedbacks_refine.items():
                feedback_with_refinement[ft] = fb[i]
            log[i].append({"attempt": n_attempts, "solution_curr": solution, "solution_fixed": solution_fixed, "feedback": feedback, "feedback_with_refinement": feedback_with_refinement})

        if not any(any(feedback_retry) for feedback_retry in feedbacks_retry):
            break

        solutions = [solution_fixed for solution_fixed in solutions_fixed]

        n_attempts += 1
        iter_end = time.time()
        logger.write(f"Iteration {n_attempts} took {(iter_end - iter_start)/60}minutes\n")
        logger.write(f"writing intermediate results\n")
        results = []
        for j, row in enumerate(data.iterrows()):
            row_copy = row[-1].to_dict()
            row_copy["run_logs"] = log[j]
            row_copy["generated_answer_direct"] = log[j][0]["solution_curr"]
            row_copy["generated_answer_ours"] = log[j][-1]["solution_fixed"]
            results.append(row_copy)
    
        pd.DataFrame(results).to_json(outfile, orient="records", lines=True)

    return log


def fix_gsm(gsm_task_file: str, prompt_dir: str, max_attempts: int, outfile: str, temperature: float, feedback_types: str, engine: str, batch_size: int = 5, gpus: str = "0,1", summarize_fb: bool = False, early_stop: bool = False, debug: bool = False):

    # prepare feedback modules

    df = pd.read_json(gsm_task_file, lines=True, orient="records")
    if debug:
        df = df[:5]
    df["run_logs"] = [None] * len(df)
    results = []
    # loop over number of attempts instead of number of datapoints to use async calls
    run_logs = iterative_gsm(outfile=outfile, data=df, prompt_dir=prompt_dir, max_attempts=max_attempts, feedback_types=feedback_types, engine=engine, temperature=temperature, batch_size=batch_size, gpus=gpus, summarize_fb=summarize_fb, early_stop=early_stop, debug=debug)

    for j, row in enumerate(df.iterrows()):
        row_copy = row[-1].to_dict()
        row_copy["run_logs"] = run_logs[j]
        row_copy["generated_answer_direct"] = run_logs[j][0]["solution_curr"]
        row_copy["generated_answer_ours"] = run_logs[j][-1]["solution_fixed"]
        results.append(row_copy)
    
    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    return results

def test():
    import json

    
    with open("/tmp/debug_gsm.jsonl", "w") as fout:
        questions = ["Milo is making a mosaic with chips of glass. It takes twelve glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?", "Kelly is grocery shopping at a supermarket and is making sure she has enough in her budget for the items in her cart. Her 5 packs of bacon cost $10 in total and she has 6 packets of chicken which each cost twice as much as a pack of bacon. She also has 3 packs of strawberries, priced at $4 each, and 7 packs of apples, each priced at half the price of a pack of strawberries. If Kelly’s budget is $65 then how much money, in dollars, does she have left in her budget?"]
        for q in questions:
            fout.write(json.dumps({"input": q}) + "\n")
        
    logs = fix_gsm(gsm_task_file='/tmp/debug_gsm.jsonl', max_attempts=3, outfile='/tmp/test.jsonl', temperature=0.7, feedback_types='variable_naming, missing_step, logical', engine='text-davinci-003')
    for i, log in enumerate(logs):
        print(log["generated_answer_ours"])
        print(log["generated_answer_direct"])


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--gsm_task_file", type=str, default="data/gsm/gsmic_mixed_1_irc.jsonl")
    args.add_argument("--max_attempts", type=int, default=1)
    args.add_argument("--save_dir", type=str, default="outputs/gsm_maf")
    args.add_argument("--exp_label", type=str, default="gsmic_mixed_1_irc_amaf_init_sequential_listwise_temp")
    args.add_argument("--feedback_types", type=str, default="variable_naming, missing_step, logical")
    args.add_argument("--prompt_dir", type=str, default="prompt/gsm_maf")
    args.add_argument("--temperature", type=float, default=0.7)
    args.add_argument("--summarize_fb", action="store_true", default=False)
    args.add_argument("--early_stop", action="store_true", default=False)
    args.add_argument("--engine", type=str, default="text-davinci-003", choices=OPENAI_ENGINES + OS_ENGINES)
    args.add_argument("--gpus", type=str, default="0,1")
    args.add_argument("--batch_size", type=int, default=5)
    args.add_argument("--debug", action="store_true", default=False)
    args = args.parse_args()
    args.outdir = os.path.join(args.save_dir, f"{args.exp_label}.temp_{args.temperature}.engine_{args.engine}")
    print(args.save_dir)
    os.makedirs(args.outdir, exist_ok=True)
    if args.debug:
        args.outfile = os.path.join(args.outdir, f"results_debug.jsonl")
    else:
        args.outfile = os.path.join(args.outdir, f"results.jsonl")    
    
    _logger = Logger(os.path.join(args.outdir, f"args.txt"))

    print('=====Input Arguments=====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        logger = Logger(f"/tmp/test.log.txt")
        test()
    else:
        args = parse_args()
        logger = Logger(os.path.join(args.outdir, f"log.txt"))
        fix_gsm(gsm_task_file=args.gsm_task_file, prompt_dir=args.prompt_dir, max_attempts=args.max_attempts, outfile=args.outfile, temperature=args.temperature, feedback_types = args.feedback_types, engine=args.engine, batch_size=args.batch_size, gpus=args.gpus, summarize_fb=args.summarize_fb, early_stop=args.early_stop, debug=args.debug)