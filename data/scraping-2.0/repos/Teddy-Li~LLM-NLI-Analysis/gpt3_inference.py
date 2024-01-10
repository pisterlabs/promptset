import json
from collections import Counter

from utils import load_general_entries, load_typed_general_entries, negate, wrap_prompt_completion, wrap_prompt_chat, \
    find_best_f_beta_from_curve, print_metrics, acquire_in_context_examples, get_gpt_template
import argparse
import openai
import os
import random
import sys
import time
import math
from typing import List, Tuple

import matplotlib.pyplot as plt



# INFERENCE_OPTION_STR_TRINARY = "\nA) Entailment\nB) Neutral\nC) Contradiction\nAnswer:"
# KNOWLEDGE_OPTION_STR_TRINARY = "\nA) True\nB) Unknown\nC) False\nAnswer:"
# INFERENCE_OPTION_STR_BINARY = " Is this True or False?\nA) True\nB) False\nAnswer:"
# KNOWLEDGE_OPTION_STR_BINARY = " Is this True or False?\nA) True\nB) False\nAnswer:"

chat_models = ['gpt-4', 'gpt-4-0314', 'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301']


def get_gpt3_output(prompt: str, model_name: str = "text-davinci-003", max_tokens: int = 32, temperature: float = 0.0,
                    top_p: float = 1.0, use_binary_options: bool = False, debug: bool = False) -> Tuple[str, float, str]:
    def option_matcher(output: str, char: str) -> bool:
        if output == char:
            return True
        elif output == char.lower():
            return True
        elif output.startswith(char + ')'):
            return True
        elif output.startswith(char + ' '):
            return True
        elif output.startswith(char + '.'):
            return True
        elif output.startswith(char + '-'):
            return True
        else:
            return False

    if args.dry_run:
        scr = random.random()
        label = 'A' if scr > 0.5 else 'B'
        return label, scr, None

    if model_name in chat_models:
        prompt_dict = wrap_prompt_chat(prompt, model_name, max_tokens, temperature, top_p)
    else:
        prompt_dict = wrap_prompt_completion(prompt, model_name, max_tokens, temperature, top_p)
    response = None
    for i in range(3):
        try:
            if model_name in ['gpt-4', 'gpt-4-0314', 'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301']:
                response = openai.ChatCompletion.create(**prompt_dict)
            else:
                response = openai.Completion.create(**prompt_dict)
            time.sleep(args.sleep_after_query)
            break
        except Exception as e:
            print(f"Error: {e}")
            if i == 4:
                pass
            else:
                time.sleep(args.sleep_after_query)
                print(f"Retrying...")
                continue

    if model_name in chat_models:
        if response is None:
            print(f"Error: response is None", file=sys.stderr)
            return 'B', 0.0, response
        else:
            ret_text = response['choices'][0]['message']['content']
            # print(f"Returned text: {ret_text}")

            if option_matcher(ret_text, 'A'):
                return 'A', 1.0, response
            elif option_matcher(ret_text, 'B'):
                return 'B', 0.0, response
            elif option_matcher(ret_text, 'C'):
                return 'C', 0.0, response
            else:
                print(f"Error: response is not a boolean: {ret_text}; regarding it as a False.", file=sys.stderr)
                return 'B', 0.0, response
    else:
        if response is not None:
            answer = response['choices'][0]['text'].strip(' ')
            if response['choices'][0]['logprobs']['tokens'][0].strip() == ':':
                logprobs_first_token = response['choices'][0]['logprobs']['tokens'][1]
            else:
                logprobs_first_token = response['choices'][0]['logprobs']['tokens'][0]
            if logprobs_first_token.strip().lower() not in ['a', 'b', 'c']:
                print(f"Error in logprobs_first_token: {logprobs_first_token}", file=sys.stderr)
                pass
            logprob = response['choices'][0]['logprobs']['token_logprobs'][0]
        else:
            answer = None
            logprobs_first_token = None
            logprob = None

        if debug:
            print(answer)
        if answer is None:
            return 'B', 0.0, response
        elif option_matcher(answer, 'A'):
            # print("!")
            assert 0 < math.exp(logprob) < 1
            effective_scr = 0.5 + 0.5*math.exp(logprob)
            return 'A', effective_scr, response
        elif use_binary_options and option_matcher(answer, 'B'):
            assert 0 < math.exp(logprob) < 1
            effective_scr = 0.5 - 0.5 * math.exp(logprob)
            return 'B', effective_scr, response
        elif (not use_binary_options) and (option_matcher(answer, 'B') or option_matcher(answer, 'C')):
            assert 0 < math.exp(logprob) < 1
            effective_scr = 0.5 - 0.5 * math.exp(logprob)
            if option_matcher(answer, 'B'):
                return 'B', effective_scr, response
            elif option_matcher(answer, 'C'):
                return 'C', effective_scr, response
            else:
                raise AssertionError
        else:
            print(f"Unexpected answer for binary_options={use_binary_options}: {answer}", file=sys.stderr)
            return 'B', 0.0, response


def vote(answers: List[bool]):
    return sum(answers) > len(answers) / 2


def retrieve_results_main(args):
    if args.hypothesis_only:
        sent_template_activate_flags = [True]
        sent_template_to_test = [
                {'s': '{hyp}.', 'do_neg': False}
        ]
    else:
        if args.tplt_id is not None:
            sent_template_activate_flags = [False] * 8
            sent_template_activate_flags[args.tplt_id] = True
        else:
            # sent_template_activate_flags = [False, True, False, False, False, False, False, False]
            # sent_template_activate_flags = [False, False, False, False, True, False, False, False]
            sent_template_activate_flags = [True, True, False, True, True, False, False, False]
            # sent_template_activate_flags = [True, True, True, True, True, True, True, True]
        sent_template_to_test = [
            {'s': "{prm}, which means that {hyp}.", 'do_neg': False},
            {'s': "If {prm}, then {hyp}.", 'do_neg': False},
            {'s': "{hyp}, because {prm}.", 'do_neg': False},
            {'s': "{prm}, so {hyp}.", 'do_neg': False},
            {'s': "{prm} entails {hyp}.", 'do_neg': False},
            {'s': "It is not the case that {hyp}, let alone {prm}.", 'do_neg': False},
            {'s': "{prm}, because {hyp}.", 'do_neg': True},
            {'s': "{hyp}, which means that {prm}.", 'do_neg': True},
        ]
    sent_template_to_test = [x for x, y in zip(sent_template_to_test, sent_template_activate_flags) if y]
    assert args.num_templates == len(sent_template_to_test)

    openai.organization = os.getenv('OPENAI_ORG_ID')
    openai.api_key = os.getenv('OPENAI_API_KEY')

    if args.use_plhr in ['original', 'xy', 'shuffled', 'randprem-orig', 'lowfreq', 'highfreq']:
        prem_hyp_pairs = load_general_entries(args.infn_for_eval)  # these are the premise-hypothesis pairs that are True Entailments
    elif args.use_plhr in ['type', 'randprem-type']:
        prem_hyp_pairs = load_typed_general_entries(args.infn_for_eval)
    else:
        raise AssertionError(f"Unknown use_plhr value: {args.use_plhr}")

    preds = [[] for x in range(args.num_templates+3)]  # the +2 are for voting and maximum
    golds = [[] for x in range(args.num_templates+3)]  # the +2 are for voting and maximum
    responses = [[] for x in range(args.num_templates)]

    ready_entries = []
    try:
        ref_fn = args.res_fn+'_ref'
        ref_fp = open(ref_fn, 'r', encoding='utf-8')
        for line in ref_fp:
            if len(line) < 2:
                continue
            item = json.loads(line)
            ready_entries.append(item)
        ref_fp.close()
        print(f"Loaded {len(ready_entries)} entries from {args.res_fn+'_ref'}")
    except FileNotFoundError:
        print(f"File {args.res_fn+'_ref'} not found, will start from scratch.")

    ofp = open(args.res_fn, 'w', encoding='utf-8')
    ready_count = 0
    print_flag = True
    start_t = time.time()

    # For each premise-hypothesis pair, get the templates and score them with the model;
    # let the 5 templates vote on which one is better.
    for ent_idx, (prem, hyp, lbl, aligned_flag) in enumerate(prem_hyp_pairs):
        if ent_idx % 5 == 0:
            curr_t = time.time()
            durr = curr_t - start_t
            print(f'Processing entry {ent_idx} of {len(prem_hyp_pairs)}; durr: {durr//60}m {durr%60}s;')

        if lbl == 'True':
            lbl = True
        elif lbl == 'False':
            lbl = False
        else:
            raise AssertionError(f"Unknown label: {lbl}")

        ready_found = False

        heu_ent = ready_entries[ent_idx] if ent_idx < len(ready_entries) else None
        if heu_ent is not None and heu_ent['premise'] == prem and heu_ent['hypothesis'] == hyp:
            ready_ent = heu_ent
            ready_found = True
        else:
            ready_ent = None
            for ready_ent in ready_entries:
                if prem == ready_ent['premise'] and hyp == ready_ent['hypothesis']:
                    ready_found = True
                    break
        if ready_found is True and lbl == ready_ent['gold']:
            ready_found = True
            ready_count += 1
            print(f"Ready entry found for {prem} and {hyp}: cnt: {ready_count};")
            for i in range(args.num_templates):
                preds[i].append(ready_ent['preds'][i])
            binarized_preds = [x > 0.5 for x in ready_ent['preds']]
            preds[args.num_templates].append(vote(binarized_preds))
            preds[args.num_templates+1].append(any(binarized_preds))
            preds[args.num_templates+2].append(all(binarized_preds))
            for i in range(args.num_templates+3):
                golds[i].append(ready_ent['gold'])
            ofp.write(json.dumps(ready_ent, ensure_ascii=False) + '\n')
            continue
        elif ready_found is True:
            print(f"Ready entry found for {prem} and {hyp}, but the gold label is different: {ready_ent['gold']} vs. {lbl};")
            pass
        else:
            pass

        entry_preds = []
        entry_preds_binarized = []
        entry_preds_tokenized = []
        for tplt_idx in range(args.num_templates):
            if args.hypothesis_only:
                prem = None
                single_statement = 'h'
            else:
                single_statement = None
            curr_t = get_gpt_template(prem, hyp, aligned=aligned_flag, use_plhr=args.use_plhr, in_context=args.in_context,
                                      tplt_fmt=sent_template_to_test[tplt_idx]['s'],
                                      do_neg=sent_template_to_test[tplt_idx]['do_neg'], use_binary_options=args.use_binary_options,
                                      single_statement=single_statement, rev_hyp_args=args.rev_hyp_args,
                                      has_instruction=args.instruction)
            if args.debug or print_flag:
                print(f"Current prompt:")
                print(curr_t)
                print_flag = False
            curr_res, curr_scr, response = get_gpt3_output(curr_t, args.model_name, max_tokens=args.max_tokens,
                                                           temperature=args.temperature,
                                                           use_binary_options=args.use_binary_options, debug=args.debug)
            responses[tplt_idx].append(response)
            assert isinstance(curr_res, str) and isinstance(curr_scr, float)
            assert curr_res in ['A', 'B', 'C']
            preds[tplt_idx].append(curr_scr)  # here the scr > 0.5 means binary-True, and < 0.5 means binary-False
            entry_preds_tokenized.append(curr_res)
            entry_preds_binarized.append(True if curr_res == 'A' else False)
            entry_preds.append(curr_scr)
            if args.sleep_after_query > 0:
                time.sleep(args.sleep_after_query)
        preds[args.num_templates].append(vote(entry_preds_binarized))
        preds[args.num_templates+1].append(any(entry_preds_binarized))
        preds[args.num_templates+2].append(all(entry_preds_binarized))
        for i in range(args.num_templates+3):
            golds[i].append(lbl)

        out_item = {
            'premise': prem,
            'hypothesis': hyp,
            'preds': entry_preds,
            'preds_tokenized': entry_preds_tokenized,
            'gold': lbl,
        }
        ofp.write(json.dumps(out_item, ensure_ascii=False) + '\n')
        time.sleep(1)

    saved_responses_fn = args.res_fn.replace('.json', '__response.json')
    with open(saved_responses_fn, 'w', encoding='utf-8') as saved_responses_fp:
        json.dump(responses, saved_responses_fp, indent=4)

    for tplt_idx in range(args.num_templates+3):
        # Calculate the binary scores
        if tplt_idx == args.num_templates:
            print(f"Voting:")
        elif tplt_idx == args.num_templates+1:
            print(f"Any:")
        elif tplt_idx == args.num_templates+2:
            print(f"Consensus:")
        else:
            print(f"Template {tplt_idx}: {sent_template_to_test[tplt_idx]}")
        print(f"Using placeholders for the subjects and objects? {args.use_plhr}")

        # Calculate the precision-recall curve
        print_metrics(golds[tplt_idx], preds[tplt_idx], legend_str=f"{tplt_idx}", beta=args.beta)
        ofp.close()
        print(f"Finished! Results written to {args.res_fn}.")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision Recall Curves")
    plt.legend()
    plt.draw()
    # plt.show()
    assert args.res_fn.endswith('.json')
    plt.savefig(f"{args.res_fn}".replace('.json', '.png'))

    if args.subset == 'full':
        print(f"Also doing evaluation on the directional subset:")
        try:
            get_scr_from_full_result(args, dirscr=True)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Skipping directional subset evaluation.")


def get_scr_from_full_result(args, dirscr: bool):
    banned_template_ids = [5, 6, 7]  # These "banned templates" are effective only when calculating benchmark scores from raw results.
    if dirscr:
        diridx_fpath = f'./levyholt_files/dir_files/with_original/{args.split}_idxes.json'
        with open(diridx_fpath, 'r', encoding='utf-8') as diridx_fp:
            diridxes = json.load(diridx_fp)
    else:
        diridxes = None

    if args.use_plhr == 'original':
        order_str = '_entord'
    elif args.use_plhr in ['type', 'shuffled', 'pseudoents', 'randprem-type', 'randprem-orig',
                           'lowfreq', 'highfreq']:
        order_str = '_ordered'
    else:
        raise AssertionError
    inclusion_flags_fpath = f'./levyholt_files/dir_files/with_original/{args.split}_inclusion_flags{order_str}.json'
    with open(inclusion_flags_fpath, 'r', encoding='utf-8') as inclusion_flags_fp:
        inclusion_flags = json.load(inclusion_flags_fp)

    full_results = []
    try:
        with open(args.res_fn, 'r', encoding='utf-8') as res_fp:
            for line in res_fp:
                full_results.append(json.loads(line))
    except FileNotFoundError as e:
        with open(args.res_fn.replace('/results/', '/results_2/'), 'r', encoding='utf-8') as res_fp:
            for line in res_fp:
                full_results.append(json.loads(line))
    assert len(full_results) == len(inclusion_flags)

    preds = [[] for x in range(args.num_templates + 3)]  # the +2 are for voting and maximum
    golds = [[] for x in range(args.num_templates + 3)]  # the +2 are for voting and maximum
    pred_tokens = [[] for x in range(args.num_templates)]  # the +2 are for voting and maximum
    for ridx, (res_entry, i_flag) in enumerate(zip(full_results, inclusion_flags)):
        if diridxes is not None and ridx not in diridxes:
            continue
        if args.inclusion_subset == 'yes' and i_flag == False:
            continue
        elif args.inclusion_subset == 'no' and i_flag == True:
            continue

        eligible_preds = []
        for tplt_idx in range(args.num_templates):
            if tplt_idx in banned_template_ids:
                continue
            preds[tplt_idx].append(res_entry['preds'][tplt_idx])
            eligible_preds.append(res_entry['preds'][tplt_idx])
            if 'preds_tokenized' in res_entry:
                pred_tokens[tplt_idx].append(res_entry['preds_tokenized'][tplt_idx])
            else:
                pass
        eligible_preds_binarized = [x > 0.5 for x in eligible_preds]
        preds[args.num_templates].append(vote(eligible_preds_binarized))
        preds[args.num_templates+1].append(any(eligible_preds_binarized))
        preds[args.num_templates+2].append(all(eligible_preds_binarized))
        for i in range(args.num_templates+3):
            golds[i].append(res_entry['gold'])

    for tplt_idx in range(args.num_templates+3):
        if tplt_idx in banned_template_ids:
            continue
        # Calculate the binary scores
        curr_tplt_binarized_preds = [x > 0.5 for x in preds[tplt_idx]]
        if tplt_idx < len(pred_tokens) and len(pred_tokens[tplt_idx]) > 0:
            print(f"Template {tplt_idx} Predicted label distribution:")
            curr_tplt_pred_tokens = pred_tokens[tplt_idx]
            assert all([x in ['A', 'B', 'C'] for x in curr_tplt_pred_tokens])
            a_cnt = sum([1 for x in curr_tplt_pred_tokens if x == 'A'])
            b_cnt = sum([1 for x in curr_tplt_pred_tokens if x == 'B'])
            c_cnt = sum([1 for x in curr_tplt_pred_tokens if x == 'C'])
            total_cnt = len(curr_tplt_pred_tokens)
            print(f"Pred tokenized: A: {a_cnt} ({a_cnt/total_cnt:.4f}), B: {b_cnt} ({b_cnt/total_cnt:.4f}), C: {c_cnt} ({c_cnt/total_cnt:.4f}); Total: {total_cnt}")
        else:
            print(f"Predicted label distribution unavailable.")

        if tplt_idx == args.num_templates:
            print(f"Voting:")
        elif tplt_idx == args.num_templates + 1:
            print(f"Any:")
        elif tplt_idx == args.num_templates + 2:
            print(f"Consensus:")
        else:
            print(f"Template {tplt_idx}:")
        print(f"Using placeholders for the subjects and objects? {args.use_plhr}")

        # Calculate the precision-recall curve
        print_metrics(golds[tplt_idx], preds[tplt_idx], f"Template {tplt_idx}", args.beta)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision Recall Curves")
    plt.legend()
    plt.draw()
    plt.show()
    assert args.res_fn.endswith('.json')
    plt.savefig(f"{args.res_fn}".replace('.json', f'_inc={args.inclusion_subset}.png'))


def format_data_main(args):
    if args.hypothesis_only:
        sent_template_activate_flags = [True]
        sent_template_to_test = [
                {'s': '{hyp}.', 'do_neg': False}
        ]
    else:
        if args.tplt_id is not None:
            sent_template_activate_flags = [False] * 8
            sent_template_activate_flags[args.tplt_id] = True
        else:
            # sent_template_activate_flags = [False, True, False, False, False, False, False, False]
            # sent_template_activate_flags = [False, False, False, False, True, False, False, False]
            sent_template_activate_flags = [True, True, False, True, True, False, False, False]
            # sent_template_activate_flags = [True, True, True, True, True, True, True, True]
        sent_template_to_test = [
            {'s': "{prm}, which means that {hyp}.", 'do_neg': False},
            {'s': "If {prm}, then {hyp}.", 'do_neg': False},
            {'s': "{hyp}, because {prm}.", 'do_neg': False},
            {'s': "{prm}, so {hyp}.", 'do_neg': False},
            {'s': "{prm} entails {hyp}.", 'do_neg': False},
            {'s': "It is not the case that {hyp}, let alone {prm}.", 'do_neg': False},
            {'s': "{prm}, because {hyp}.", 'do_neg': True},
            {'s': "{hyp}, which means that {prm}.", 'do_neg': True},
        ]
    sent_template_to_test = [x for x, y in zip(sent_template_to_test, sent_template_activate_flags) if y]
    assert args.num_templates == len(sent_template_to_test)

    if args.use_plhr in ['original', 'randprem-orig', 'lowfreq', 'highfreq']:
        prem_hyp_pairs = load_general_entries(args.infn_for_eval)  # these are the premise-hypothesis pairs that are True Entailments
    elif args.use_plhr in ['type', 'randprem-type']:
        prem_hyp_pairs = load_typed_general_entries(args.infn_for_eval)
    else:
        raise AssertionError(f"Unknown use_plhr value: {args.use_plhr}")

    inputs_list = [[] for i in range(args.num_templates)]

    # For each premise-hypothesis pair, get the templates and score them with the model;
    # let the 5 templates vote on which one is better.
    for ent_idx, (prem, hyp, lbl, aligned_flag) in enumerate(prem_hyp_pairs):
        if ent_idx % 1 == 0:
            print(f'Processing entry {ent_idx} of {len(prem_hyp_pairs)};')

        if lbl == 'True':
            lbl = True
        elif lbl == 'False':
            lbl = False
        else:
            raise AssertionError(f"Unknown label: {lbl}")


        for tplt_idx in range(args.num_templates):
            if args.hypothesis_only:
                prem = None
                single_statement = 'h'
            else:
                single_statement = None
            curr_t = get_gpt_template(prem, hyp, aligned=aligned_flag, use_plhr=args.use_plhr, in_context=args.in_context,
                                      tplt_fmt=sent_template_to_test[tplt_idx]['s'],
                                      do_neg=sent_template_to_test[tplt_idx]['do_neg'], use_binary_options=args.use_binary_options,
                                      single_statement=single_statement,
                                      rev_hyp_args=args.rev_hyp_args, has_instruction=args.instruction)
            inputs_list[tplt_idx].append({'in': curr_t, 'out': None, 'gold': lbl})

    with open(args.formatted_fn, 'w', encoding='utf8') as ofp:
        json.dump(inputs_list, ofp, indent=4, ensure_ascii=False)


def run_any(args):
    assert args.instantiated_in_path is not None and args.instantiated_in_path.endswith('.json')

    with open(args.instantiated_in_path, 'r', encoding='utf8') as ifp:
        inputs_list = json.load(ifp)
    openai.organization = os.getenv('OPENAI_ORG_ID')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    responses = [[] for i in range(len(inputs_list))]  # one list per template
    preds = [[] for i in range(len(inputs_list))]
    golds = []
    preds_tokenized = [[] for i in range(len(inputs_list))]
    preds_binarized = [[] for i in range(len(inputs_list))]

    ofp = open(args.instantiated_res_path, 'w', encoding='utf-8')

    total_entries = len(inputs_list[0])
    for eidx in range(total_entries):
        if eidx % 1 == 0:
            print(f"Processing entry {eidx} / {total_entries};")

        entry_preds = []
        entry_preds_tokenized = []
        entry_inputs = []
        lbl = inputs_list[0][eidx]['label']
        golds.append(lbl)
        for tplt_idx in range(len(inputs_list)):
            curr_item = inputs_list[tplt_idx][eidx]
            curr_t = curr_item['in']
            entry_inputs.append(curr_t)
            assert curr_item['label'] == lbl
            assert isinstance(curr_t, str)
            curr_res, curr_scr, response = get_gpt3_output(curr_t, args.model_name, max_tokens=args.max_tokens,
                                                           temperature=args.temperature,
                                                           use_binary_options=args.use_binary_options, debug=args.debug)
            responses[tplt_idx].append(response)
            assert isinstance(curr_res, str) and isinstance(curr_scr, float)
            assert curr_res in ['A', 'B', 'C']
            preds[tplt_idx].append(curr_scr)  # here the scr > 0.5 means binary-True, and < 0.5 means binary-False
            preds_tokenized[tplt_idx].append(curr_res)
            preds_binarized[tplt_idx].append(True if curr_res == 'A' else False)
            entry_preds.append(curr_scr)
            entry_preds_tokenized.append(curr_res)
            if args.sleep_after_query > 0:
                time.sleep(args.sleep_after_query)

        out_item = {
            'inputs': entry_inputs,
            'preds': entry_preds,
            'preds_tokenized': entry_preds_tokenized,
            'gold': lbl,
        }
        ofp.write(json.dumps(out_item, ensure_ascii=False) + '\n')
        time.sleep(2)
    print(f"Results written to {args.res_fn}.")
    ofp.close()

    # Now compute the accuracy of each template
    for tplt_idx in range(len(inputs_list)):
        # Calculate the binary scores
        print_metrics(golds, preds[tplt_idx], f"Template {tplt_idx}", args.beta)
        pred_a_cnt = Counter(preds_tokenized[tplt_idx])['A']
        pred_b_cnt = Counter(preds_tokenized[tplt_idx])['B']
        pred_c_cnt = Counter(preds_tokenized[tplt_idx])['C']
        print(f"Prediction distribution: A: {pred_a_cnt}, B: {pred_b_cnt}, C: {pred_c_cnt}; Total: {len(preds_tokenized[tplt_idx])}")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision Recall Curves")
    plt.legend()
    plt.draw()
    # plt.show()
    assert args.res_fn.endswith('.json')
    plt.savefig(f"{args.res_fn}".replace('.json', '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_fn', type=str,
                        default='./levyholt_files/%s_files/with_original/%s.txt')
    parser.add_argument('--typed_in_fn', type=str,
                        default='./levyholt_files/%s_files/with_type/%s%s.txt')  # from '../../entgraph_eval/gfiles/ent/test_dir%s.txt'
    parser.add_argument('--shuffled_in_fn', type=str,
                        default='./levyholt_files/%s_files/with_shuffled_entities/%s.txt')
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--max_tokens', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--results_root', type=str, default='./results/gpt_results')
    parser.add_argument('--res_fn', type=str, default='gpt3_%s_res_%s_text_%s_%s_icl=%s%s%s_%d%s.json')
    parser.add_argument('--formatted_fn', type=str, default='./formatted/gpt4_formin_%s_%s_%s_icl=%s%s%s_%d.json')
    parser.add_argument('--use_plhr', type=str, default='original')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--in_context', type=str, default='none')
    parser.add_argument('--num_templates', type=int, default=4)
    parser.add_argument('--hypothesis-only', action='store_true')
    parser.add_argument('--subset', type=str, default='dir', choices=['dir', 'full'])
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--task', type=str, default='query')
    parser.add_argument('--dry-run', action='store_true')  # will not call the actual API; instead use random fake data
    parser.add_argument('--rev-hyp-args', action='store_true')
    parser.add_argument('--use-binary-options', action='store_true')
    parser.add_argument('--inclusion_subset', type=str, default='any', choices=['any', 'yes', 'no'])
    parser.add_argument('--beta', type=float, default=0.5, help='beta for F-score.')
    parser.add_argument('--tplt_id', type=int, default=None)

    parser.add_argument('--res_suffix', type=str, default='')
    parser.add_argument('--sleep_after_query', type=float, default=1)

    parser.add_argument('--instruction', action='store_true')

    parser.add_argument('--instantiated_in_path', type=str, default=None)
    parser.add_argument('--instantiated_res_path', type=str, default=None)


    args = parser.parse_args()
    print(args)
    assert args.use_plhr in ['original', 'type', 'randprem-type', 'randprem-orig', 'lowfreq', 'highfreq']
    assert not (args.hypothesis_only and (args.in_context not in ['none', 'lbl'])), 'Not Implemented: ICL with Explanations with Hypothesis-only baseline'
    # assert not (args.hypothesis_only and (args.use_plhr !=  'original')), 'Not Implemented: argument replacements with Hypothesis-only baseline'
    assert args.inclusion_subset in ['any', 'yes', 'no']

    binary_str = '_binary' if args.use_binary_options else '_trinary'
    instruct_str = '_instruct' if args.instruction else ''
    hyponly_str = '_hyponly' if args.hypothesis_only else ''
    args.res_fn = args.res_fn % (args.model_name, args.subset, args.split, args.use_plhr, args.in_context, binary_str, instruct_str, args.num_templates, hyponly_str)
    args.res_fn = os.path.join(args.results_root, args.res_fn)
    args.formatted_fn = args.formatted_fn % (args.subset, args.split, args.use_plhr, args.in_context, binary_str, instruct_str, args.num_templates)
    if args.rev_hyp_args:
        args.res_fn = args.res_fn.replace('.json', '_rev-hyp-args.json')

    if args.use_plhr in ['original', 'xy']:
        args.infn_for_eval = args.in_fn % (args.subset, args.split)
    elif args.use_plhr == 'shuffled':
        args.infn_for_eval = args.shuffled_in_fn % (args.subset, args.split)
    elif args.use_plhr == 'type':
        args.infn_for_eval = args.typed_in_fn % (args.subset, args.split, '%s')
    elif args.use_plhr == 'randprem-orig':
        args.infn_for_eval = f'./levyholt_files/{args.subset}_files/randprem_files/test_randprem.txt'
    elif args.use_plhr == 'randprem-type':
        args.infn_for_eval = f'./levyholt_files/{args.subset}_files/randprem_files/test_randprem%s.txt'
    elif args.use_plhr == 'lowfreq':
        args.infn_for_eval = f'./levyholt_files/{args.subset}_files/swapped_entities/{args.split}_bottom0.05.txt'
    elif args.use_plhr == 'highfreq':
        args.infn_for_eval = f'./levyholt_files/{args.subset}_files/swapped_entities/{args.split}_top0.05.txt'
    else:
        raise NotImplementedError
    print(f"Evaluating {args.infn_for_eval} with model {args.model_name}, and saving results to {args.res_fn}")

    if args.task == 'benchmark':
        print(f"Getting scores for the full dataset:")
        get_scr_from_full_result(args, dirscr=False)
    elif args.task == 'query':
        retrieve_results_main(args)
    elif args.task == 'data':
        format_data_main(args)
    elif args.task == 'run_any':
        args.instantiated_in_path = args.instantiated_in_path % (args.split, args.use_plhr)
        args.instantiated_res_path = args.instantiated_res_path % (args.model_name, args.split, args.use_plhr)
        print(
            f"Running inference for input: {args.instantiated_in_path}; saving results to {args.instantiated_res_path}")
        run_any(args)
    else:
        raise ValueError()
