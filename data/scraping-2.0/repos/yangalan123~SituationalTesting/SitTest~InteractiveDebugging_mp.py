import argparse
import copy
import glob
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor, Future
from functools import partial

import loguru
import numpy as np
from SitTest.utils import (
    process_gt_answer_response,
    compose_chat_style_wordload_from_normal_workload,
    extract_states
)
from tqdm import tqdm

from APIServer import OpenAIServer
from Const import pricing
from Data.SimpleBoxOpeningEnv.action_templates import extract_arguments_from_single_action, \
    extract_arguments_from_multi_actions
from Data.SimpleBoxOpeningEnv.instruction_gen_pipeline import extract_num_boxes_and_num_keys, \
    extract_logic_functor_for_box_and_keys
from utils import get_gpt_tokenizer, compose_answer_from_status



def parse_args():
    parser = argparse.ArgumentParser(description='Interactive Debugging')
    parser.add_argument('--model_type', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--process_root_dir', type=str, default="../GPT3Output")
    parser.add_argument('--output_root_dir', type=str,
                        default="../GPT3Output_InteractiveDebugging/")
    parser.add_argument('--log_root_dir', type=str,
                        default="../GPT3Output_InteractiveDebugging/{}/log_dir/")
    parser.add_argument('--accounting_only', action='store_true', help='whether to only do accounting')
    parser.add_argument("--f1", action="store_true", help="whether to use f1 as the metric")
    parser.add_argument("--ignore_reverse_gt_val", action="store_true",
                        help="whether to ignore reverse gt val (note that in early version of run.py, we did not reverse gt val, so we need to ignore it)")
    parser.add_argument("--max_workers", type=int, default=5, help="max number of workers")
    parser.add_argument("--chat_style_probing", action="store_true", help="whether use chat style probing")
    parser.add_argument("--action_style_probing", action="store_true", help="whether use action style probing")
    parser.add_argument("--hints", action="store_true", help="whether use hints")
    args = parser.parse_args()
    # args.model_type = os.path.basename(args.process_root_dir)
    args.process_root_dir = os.path.join(args.process_root_dir, args.model_type)
    # see whether process_root_dir is valid
    if not os.path.exists(args.process_root_dir):
        raise ValueError(f"process_root_dir {args.process_root_dir} does not exist")
    # see how many directories are in process_root_dir
    dirs = glob.glob(os.path.join(args.process_root_dir, "*"))
    if len(dirs) == 0:
        raise ValueError(f"process_root_dir {args.process_root_dir} is empty")
    try:
        encoding = get_gpt_tokenizer(args.model_type)
    except:
        raise ValueError(f"Model type {args.model_type} not supported")
    args.output_root_dir = os.path.join(args.output_root_dir, args.model_type)
    os.makedirs(args.output_root_dir, exist_ok=True)
    args.log_root_dir = args.log_root_dir.format(args.model_type)
    os.makedirs(args.log_root_dir, exist_ok=True)
    args.exp_name = "running_log" if not args.accounting_only else "accounting_log"
    if args.hints:
        args.exp_name += "_hints"

    return args


def process_file(file, args, logger, agent: OpenAIServer):
    response = pickle.load(open(file, "rb"))
    workload = response["workload"]
    env = response["env"] if "env" in response else None
    lines = workload.split("\n")
    box_name, key_name = None, None
    num_boxes, num_keys = None, None
    logic_functor = None
    all_states = dict()
    all_histories = []
    all_prefixes = []
    init_flag = False
    init_state = None
    action_buf = []
    for line_i, line in enumerate(lines):
        if "Instruction" in line:
            if env is None:
                # we have to use our best guess here from instruction
                # step-1: solve the number of boxes and keys first
                num_boxes, num_keys = extract_num_boxes_and_num_keys(line)
                # step-2: solve the logic functor
                logic_functor = extract_logic_functor_for_box_and_keys(line)
            else:
                num_boxes, num_keys = env.num_boxes, env.num_keys
                logic_functor = env.logic_functor

        if "Step-0" in line:
            init_flag = True
            continue
        if init_flag:
            # load the initial state, each state is expressed as logic_functor['box'](box_name-box_i)=True/False, or logic_functor['key'](key_name-key_i)=True/False
            # write a program to use regex to extract the initial state
            # extract_states(line, logic_functor['box'], )
            init_state = copy.copy(lines[line_i + 1])
            init_flag = False
            continue

        if args.action_style_probing:

            if not init_flag and "Step" in line and "Step-0" not in line:
                step, box_is, key_is, _box_name, _key_name = extract_arguments_from_multi_actions(line, env)
                if box_name is None and key_name is None:
                    box_name = _box_name
                    key_name = _key_name
                    box_init_states = extract_states(init_state, logic_functor['box'], box_name)
                    key_init_states = extract_states(init_state, logic_functor['key'], key_name)
                    all_states = {**box_init_states, **key_init_states}
                    if args.reverse_gt_val:
                        for key in all_states:
                            all_states[key] = not all_states[key]
                    all_histories.append(copy.deepcopy(all_states))
                    all_prefixes.append("\n".join(lines[:line_i]))
                assert box_name == _box_name and key_name == _key_name
                for box_i, key_i in zip(box_is, key_is):
                    try:
                        all_states["{}-{}".format(box_name, box_i)] = not all_states["{}-{}".format(box_name, box_i)]
                        all_states["{}-{}".format(key_name, key_i)] = not all_states["{}-{}".format(key_name, key_i)]
                    except:
                        print(all_states, box_name, key_name, init_state, logic_functor, line)
                        exit()
                all_histories.append(copy.deepcopy(all_states))
                all_prefixes.append("\n".join(lines[:line_i + 1]))
                action_buf.append([box_is, key_is, copy.deepcopy(all_states)])

        else:
            if "Step" in line and "Step-0" not in line:
                step, box_i, key_i, _box_name, _key_name = extract_arguments_from_single_action(line, env)
                if box_name is None and key_name is None:
                    box_name = _box_name
                    key_name = _key_name
                    box_init_states = extract_states(init_state, logic_functor['box'], box_name)
                    key_init_states = extract_states(init_state, logic_functor['key'], key_name)
                    all_states = {**box_init_states, **key_init_states}
                    if args.reverse_gt_val:
                        for key in all_states:
                            all_states[key] = not all_states[key]
                    # note this is not duplicated -- this is for the step-0
                    all_histories.append(copy.deepcopy(all_states))
                    all_prefixes.append("\n".join(lines[:line_i]))
                assert box_name == _box_name and key_name == _key_name
                all_states["{}-{}".format(box_name, box_i)] = not all_states["{}-{}".format(box_name, box_i)]
                all_states["{}-{}".format(key_name, key_i)] = not all_states["{}-{}".format(key_name, key_i)]
                all_histories.append(copy.deepcopy(all_states))
                all_prefixes.append("\n".join(lines[:line_i + 1]))
                action_buf.append([[box_i, ], [key_i, ], copy.deepcopy(all_states)])

    assert num_boxes is not None and num_keys is not None and logic_functor is not None
    reconstructed_gt_answer = compose_answer_from_status(all_states, box_name, key_name, num_boxes, num_keys,
                                                         logic_functor) + "."
    gt_answer = response['gt_answer']
    _pseudo_response = copy.deepcopy(response)

    _pseudo_response['choices'][0]['message']['content'] = reconstructed_gt_answer
    _em_acc, _stat_acc, _tv_all, _parseable_acc = process_gt_answer_response(gt_answer, _pseudo_response, logger,
                                                                             reverse_gt_val=args.reverse_gt_val,
                                                                             f1=args.f1)
    assert (args.f1 and _stat_acc['f1'] == 1) or (
            not args.f1 and _stat_acc == 1), "gt_answer not reconstructed correctly: \ngt_answer: {}\nreconstructed: {}".format(
        gt_answer, reconstructed_gt_answer)
    _em_acc, _stat_acc, _tv_all, _parseable_acc = process_gt_answer_response(gt_answer, response, logger,
                                                                             reverse_gt_val=args.reverse_gt_val,
                                                                             f1=args.f1)
    ret_new_workload = []
    all_budgets = []
    hint_states = dict()
    if (not args.f1 and _stat_acc < 1) or (args.f1 and _stat_acc['f1'] < 1):
        for history_i in range(1, len(all_histories)):
            cur_history = all_histories[history_i]
            cur_prefix = all_prefixes[history_i]
            if args.hints:
                action_item = action_buf[history_i - 1]
                for box_i in action_item[0]:
                    hint_states["{}-{}".format(box_name, box_i)] = action_item[2]["{}-{}".format(box_name, box_i)]
                for key_i in action_item[1]:
                    hint_states["{}-{}".format(key_name, key_i)] = action_item[2]["{}-{}".format(key_name, key_i)]
            cur_answer = compose_answer_from_status(cur_history, box_name, key_name, num_boxes, num_keys, logic_functor)
            cur_question = compose_answer_from_status(cur_history, box_name, key_name, num_boxes, num_keys,
                                                      logic_functor, is_answer=False, give_hints=args.hints,
                                                      hint_states=hint_states)
            new_prefix = "\n".join([cur_prefix, f"Question: {cur_question}", "Answer: "])
            ret_new_workload.append([new_prefix, cur_answer])
            accounting_message_workload = new_prefix + cur_answer
            if agent.is_chat_model():
                if args.chat_style_probing:
                    accounting_message_workload = agent.prepare_chat_workload(
                        compose_chat_style_wordload_from_normal_workload(accounting_message_workload))
                else:
                    accounting_message_workload = agent.prepare_chat_workload(accounting_message_workload)
            num_tokens = agent.send_accounting(accounting_message_workload)
            pricing_type = pricing[args.model_type]
            all_budgets.append(num_tokens / pricing_type[1] * pricing_type[0])
    return ret_new_workload, all_budgets, _stat_acc


def worker(agent, worker_id, filename, target_filename, logger, args):
    _new_workload, _new_budgets, _stat_acc = process_file(filename, args, logger, agent)
    if isinstance(_stat_acc, dict):
        _stat_acc_local = _stat_acc['f1']
    else:
        _stat_acc_local = _stat_acc
    file_em_acc = 0
    file_stat_acc = 0
    file_parseable_acc = 0
    file_tv_all = []
    file_responses = []
    file_counter = 0
    for _item_i, _item in enumerate(
            tqdm(_new_workload, desc=f"Worker {worker_id}", position=1 + (worker_id) % args.max_workers, leave=True)):
        _workload, _gt_answer = _item
        # repeatedly send the workload until it is parseable
        _parseable = False
        _parseable_counter = 0
        while not _parseable and _parseable_counter < 5:
            try:
                if args.chat_style_probing:
                    response = agent.send(compose_chat_style_wordload_from_normal_workload(_workload), logprobs=5)
                else:
                    response = agent.send(_workload, logprobs=5)
                break
            except Exception as e:
                # network issue, too long context, etc.
                _parseable_counter += 1
                logger.warning(
                    f"Worker {worker_id} failed to parse workload {_workload} for {filename} for {_parseable_counter} / 5 times, Exception: {e}")
                # sleep for 0.5 second
                time.sleep(0.5)
                continue
        file_counter += 1
        _em_acc, _stat_acc, _tv_all, _parseable_acc = process_gt_answer_response(_gt_answer, response, logger,
                                                                                 reverse_gt_val=args.reverse_gt_val)
        file_em_acc += _em_acc
        file_stat_acc += _stat_acc
        file_tv_all += _tv_all
        file_parseable_acc += _parseable_acc
        response['gt_answer'] = _gt_answer
        response['workload'] = _workload
        response['source_filename'] = filename
        response['debug_item_no'] = _item_i
        if args.chat_style_probing:
            response['chat_style_workload'] = compose_chat_style_wordload_from_normal_workload(_workload)
        file_responses.append(response)

    file_em_acc /= (file_counter + 1e-10)
    file_stat_acc /= (file_counter + 1e-10)
    file_parseable_acc /= (file_counter + 1e-10)
    if len(_new_workload) > 0 and len(file_responses) > 0:
        pickle.dump([file_em_acc, file_stat_acc, file_tv_all, file_parseable_acc, _new_workload, file_responses],
                    open(target_filename, "wb"))
    return worker_id, _new_workload, _new_budgets, _stat_acc_local


def process_result(future: Future, mp_results: list, progress_bar: tqdm):
    result = future.result()
    mp_results.append(result)
    progress_bar.update(1)


if __name__ == '__main__':
    # write an argumentparser to handle process_root_dir, output_dir, and other arguments
    args = parse_args()
    logger = loguru.logger
    logger.add(f"{args.log_root_dir}/" + args.exp_name + ".log", mode='w')
    agent = OpenAIServer(args.model_type)
    all_budgets = 0
    example_flag = True
    example_count = 5
    path_search_pattern = args.process_root_dir + "/*NL_func_NL_arg*"
    if args.chat_style_probing:
        path_search_pattern = args.process_root_dir + "/*func*arg*chat_style*"
    for exp_path in glob.glob(path_search_pattern):
        if "incomp" in exp_path:
            continue
        logger.info("processing exp path: {}".format(exp_path))
        response_pickle_path = os.path.join(args.output_root_dir, args.model_type, os.path.basename(exp_path),
                                            "responses_dir")
        if args.chat_style_probing:
            response_pickle_path = os.path.join(args.output_root_dir, args.model_type, os.path.basename(exp_path),
                                                "chat_style_responses_dir")
        if args.action_style_probing:
            response_pickle_path = os.path.join(response_pickle_path, "action_style_responses_dir")
        if not os.path.exists(response_pickle_path):
            os.makedirs(response_pickle_path)
        try:
            args.reverse_gt_val = True if "cf" in exp_path and not args.ignore_reverse_gt_val else False
            filenames = glob.glob(os.path.join(exp_path, "*"))
            dispatch_filenames = []
            for filename in tqdm(filenames, position=0, leave=True, desc="dispatching workload"):
                if ".py" in filename:
                    continue
                target_filename = os.path.join(response_pickle_path, os.path.basename(filename) + ".pkl")
                if os.path.exists(target_filename):
                    try:
                        _tmp_data = pickle.load(open(target_filename, "rb"))
                        logger.info(
                            f"detected processing file {filename} has been intereactly debugged and output is at {target_filename}, checking whether it is correctly dumped")
                        if len(_tmp_data[-1]) == 0:
                            logger.info(
                                f"detected processing file {filename} has been intereactly debugged and output is at {target_filename}, but the output is empty, reprocessing")
                            os.remove(target_filename)
                            logger.warning(f"removed empty file {target_filename}")
                        else:
                            continue
                    except:
                        pass
                dispatch_filenames.append([filename, target_filename])

            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                partial_worker = partial(worker, agent=agent, logger=logger, args=args)
                progress_bar = tqdm(total=len(dispatch_filenames), position=0, leave=True, desc="processing workload")
                mp_results = []
                for worker_id, (filename, target_filename) in enumerate(dispatch_filenames):
                    future = executor.submit(partial_worker, worker_id=worker_id, filename=filename,
                                             target_filename=target_filename)
                    future.add_done_callback(partial(process_result, mp_results=mp_results, progress_bar=progress_bar))
            progress_bar.close()
            mp_results.sort(key=lambda x: x[0])

            new_workload = [x[1] for x in mp_results]
            new_budgets = [sum(x[2]) for x in mp_results]
            stat_acc = [x[3] for x in mp_results]
            logger.info("new workload size: {}".format(len(new_workload)))
            logger.info("new budgets: {}".format(sum(new_budgets)))
            logger.info("stat acc: {}".format(np.mean(stat_acc)))
            if example_flag and example_count > 0 and len(new_workload) > 0:
                logger.info("example workload: {}".format(random.sample(new_workload, 1)[0]))
                example_count -= 1
            all_budgets += sum(new_budgets)
        except Exception as e:
            logger.info("error in processing exp path: {}".format(exp_path))
            logger.info("error: {}".format(e))
    logger.info("all budgets: {}".format(all_budgets))
