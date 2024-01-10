import argparse
import copy
import glob
import os
import pickle
import random
import traceback

import loguru
import numpy as np
from SitTest.utils import (
    process_gt_answer_response,
    extract_states
)

from APIServer import OpenAIServer
from Const import pricing
from Data.SimpleBoxOpeningEnv.action_templates import extract_arguments_from_multi_actions
from Data.SimpleBoxOpeningEnv.instruction_gen_pipeline import extract_num_boxes_and_num_keys, \
    extract_logic_functor_for_box_and_keys
from utils import get_gpt_tokenizer, compose_answer_from_status


def parse_args():
    parser = argparse.ArgumentParser(description='Interactive Debugging')
    parser.add_argument('--model_type', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--process_root_dir', type=str, default="../GPT3Output")
    parser.add_argument('--output_root_dir', type=str, default="../GPT3Output_InteractiveDebugging/")
    parser.add_argument('--log_root_dir', type=str, default="../GPT3Output_InteractiveDebugging/{}/log_dir/")
    parser.add_argument('--accounting_only', action='store_true', help='whether to only do accounting')
    parser.add_argument("--f1", action="store_true", help="whether to use f1 as the metric")
    parser.add_argument("--ignore_reverse_gt_val", action="store_true",
                        help="whether to ignore reverse gt val (note that in early version of run.py, we did not reverse gt val, so we need to ignore it)")
    parser.add_argument("--use_llama", action="store_true", help="whether to use llama")
    parser.add_argument("--chat_style_probing", action="store_true", help="whether use chat style probing")
    parser.add_argument("--action_style_probing", action="store_true", help="whether use action probing")
    parser.add_argument("--action_perturbation", type=str, default="verbose",
                        help="what kind of perturbation you want to apply?")
    parser.add_argument("--prerun_max_steps", type=int, default=0, help="max steps for pre-run")
    args = parser.parse_args()
    args.process_root_dir = os.path.join(args.process_root_dir, args.model_type)
    # see whether process_root_dir is valid
    if not os.path.exists(args.process_root_dir):
        raise ValueError(f"process_root_dir {args.process_root_dir} does not exist")
    # see how many directories are in process_root_dir
    dirs = glob.glob(os.path.join(args.process_root_dir, "*"))
    if len(dirs) == 0:
        raise ValueError(f"process_root_dir {args.process_root_dir} is empty")
    if not args.use_llama:
        try:
            encoding = get_gpt_tokenizer(args.model_type)
        except:
            raise ValueError(f"Model type {args.model_type} not supported")
    args.output_root_dir = os.path.join(args.output_root_dir, args.model_type)
    os.makedirs(args.output_root_dir, exist_ok=True)
    args.log_root_dir = args.log_root_dir.format(args.model_type)
    os.makedirs(args.log_root_dir, exist_ok=True)
    args.exp_name = "running_log_f1" if not args.accounting_only else "accounting_log"

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

        if not init_flag and "Step" in line and "Step-0" not in line:
            step, box_is, key_is, _box_name, _key_name = extract_arguments_from_multi_actions(line, env)
            if box_name is None and key_name is None:
                box_name = _box_name
                key_name = _key_name
                box_init_states = extract_states(init_state, logic_functor['box'], box_name)
                key_init_states = extract_states(init_state, logic_functor['key'], key_name)
                # merge the two states
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

    assert num_boxes is not None and num_keys is not None and logic_functor is not None
    reconstructed_gt_answer = compose_answer_from_status(all_states, box_name, key_name, num_boxes, num_keys,
                                                         logic_functor) + "."
    gt_answer = response['gt_answer']
    _pseudo_response = copy.deepcopy(response)

    _pseudo_response['choices'][0]['message']['content'] = reconstructed_gt_answer
    _em_acc, _stat_acc, _tv_all, _parseable_acc = process_gt_answer_response(gt_answer, _pseudo_response, logger,
                                                                             # as we previously do not alter gt_output to reverse_gt_val in run.py, we do not need to do it here
                                                                             reverse_gt_val=False, f1=args.f1,
                                                                             is_llama=args.use_llama)
    assert (args.f1 and _stat_acc['f1'] == 1) or (
                not args.f1 and _stat_acc == 1), "gt_answer not reconstructed correctly: \ngt_answer: {}\nreconstructed: {}".format(
        gt_answer, reconstructed_gt_answer)

    _em_acc, _stat_acc, _tv_all, _parseable_acc = process_gt_answer_response(gt_answer, response, logger,
                                                                             reverse_gt_val=args.reverse_gt_val,
                                                                             f1=args.f1, is_llama=args.use_llama)
    ret_new_workload = []
    all_budgets = []
    if (not args.f1 and _stat_acc < 1) or (args.f1 and _stat_acc['f1'] < 1):
        for history_i in range(1, len(all_histories)):
            cur_history = all_histories[history_i]
            cur_prefix = all_prefixes[history_i]
            cur_answer = compose_answer_from_status(cur_history, box_name, key_name, num_boxes, num_keys, logic_functor)
            cur_question = compose_answer_from_status(cur_history, box_name, key_name, num_boxes, num_keys,
                                                      logic_functor, is_answer=False)
            new_prefix = "\n".join([cur_prefix, f"Question: {cur_question}", "Answer: "])
            ret_new_workload.append([new_prefix, cur_answer])
            accounting_message_workload = new_prefix + cur_answer
            if agent.is_chat_model():
                accounting_message_workload = agent.prepare_chat_workload(accounting_message_workload)
            num_tokens = agent.send_accounting(accounting_message_workload)
            if args.model_type in pricing:
                pricing_type = pricing[args.model_type]
            else:
                # for llama-based models
                pricing_type = [0, 1]
            all_budgets.append(num_tokens / pricing_type[1] * pricing_type[0])
    return ret_new_workload, all_budgets, _stat_acc


if __name__ == '__main__':
    # write an argumentparser to handle process_root_dir, output_dir, and other arguments
    args = parse_args()
    logger = loguru.logger
    logger.add(f"{args.log_root_dir}/" + args.exp_name + ".log", mode='w')
    agent = OpenAIServer(args.model_type, is_llama=args.use_llama)
    all_budgents = 0
    example_flag = True
    example_count = 5
    exp_results = []
    explanations = [
        # "NL Functor + NL Argument w/o init",
        "NL Functor + NL Argument w/ init",
        # "Random Functor + NL Argument w/o init",
        "Random Functor + NL Argument w/ init",
        # "NL Functor + Random Argument w/o init",
        "NL Functor + Random Argument w/ init",
        # "Random Functor + Random Argument w/o init",
        "Random Functor + Random Argument w/ init"
    ]
    num_samples = 50
    num_box = 10
    if args.use_llama:
        shot_range = range(3)
    else:
        shot_range = [2, 3, 5]
    for suffix in ["", "_cf_nl", "_cf_logic"]:
        counter = 0
        for arg_flag in ["NL", "irreg"]:
            for func_flag in ["NL", "irreg"]:
                for init_flag in ["_init", ]:
                    tmp_buf = [explanations[counter], ]
                    for shot_num in shot_range:
                        identifier = f"{num_samples}sample_{num_box}boxes_{func_flag}_func_{arg_flag}_arg_{shot_num}shot{init_flag}" \
                                     f"{suffix}{'_llama_' + args.model_type if args.use_llama else ''}{'_chat_style_probing' if args.chat_style_probing else ''}" \
                                     f"{'_action_style_probing_' + args.action_perturbation if args.action_style_probing else ''}" \
                                     f"{'_prerun_' + str(args.prerun_max_steps) if args.prerun_max_steps > 0 else ''}"
                        exp_path = os.path.join(args.process_root_dir, identifier)

                        logger.info("processing exp path: {}".format(exp_path))
                        try:
                            new_workload = []
                            new_budgets = []
                            stat_acc = []
                            args.reverse_gt_val = True if "cf" in exp_path else False
                            for filename in glob.glob(os.path.join(exp_path, "*pkl")):
                                _new_workload, _new_budgets, _stat_acc = process_file(filename, args, logger, agent)
                                new_workload.extend(_new_workload)
                                new_budgets.extend(_new_budgets)
                                if isinstance(_stat_acc, dict):
                                    stat_acc.append(_stat_acc['recall'])
                                else:
                                    stat_acc.append(_stat_acc)
                            logger.info("new workload size: {}".format(len(new_workload)))
                            logger.info("new budgets: {}".format(sum(new_budgets)))
                            logger.info("stat acc: {}".format(np.mean(stat_acc)))
                            if len(stat_acc) > 0:
                                tmp_buf.append("$" + str(int(100 * float(np.mean(stat_acc)))) + "\%$")
                            else:
                                tmp_buf.append("-")
                            if example_flag and example_count > 0 and len(new_workload) > 0:
                                logger.info("example workload: {}".format(random.sample(new_workload, 1)[0]))
                                example_count -= 1
                            all_budgents += sum(new_budgets)
                        except Exception as e:
                            logger.info("error in processing exp path: {}".format(exp_path))
                            logger.info("error: {}".format(e))
                            logger.info("traceback: {}".format(traceback.format_exc()))

                    exp_results.append("   &   ".join(tmp_buf) + "    \\\\")
                    counter += 1

        print("{}-------------------".format(suffix if len(suffix) > 0 else "original"))
    logger.info("all budgets: {}".format(all_budgents))
    logger.info("now output the exp results")
    for line in exp_results:
        logger.info(line)
    for line in exp_results:
        print(line)
