import json
from json import JSONDecodeError

from contradiction.medical_claims.token_tagging.gpt_solver.gpt_solver import GPTSolver, GPTRequester, GPTSolverFileRead, \
    get_parse_answer_texts_for_instruct, load_json_log, get_score_from_answer_spans_chat
from cpath import output_path
from misc_lib import path_join
from utils.open_ai_api import OpenAIProxy, parse_instruct_gpt_response, ENGINE_GPT_3_5, parse_chat_gpt_response


def get_mismatch_prediction_prompt_template_chat_gpt():
    instruction = "In each of the examples, " \
                  "two claims extracted from research paper abstracts will be shown. " \
                  "The given two claims seem to be contradictory as they are implying" \
                  " opposite results about the same question. " \
                  "Precisely though, the two claims may have been obtained" \
                  " for different population or intervention details " \
                  "that make it possible that both claims to be true." \
                  " We want to annotate the all words that" \
                  " express different conditions."
    format_instruction = "Select all such words for each of Claim1 and Claim2. " \
                         "Print results in a json format with the key \"claim1\" and \"claim2\""

    problem = "Claim 1: {}\nClaim 2: {}"
    return instruction + "\n" + format_instruction + "\n\n" + problem


def get_conflict_prediction_prompt_template_chat_gpt():
    instruction = "In each of the examples, " \
                  "two claims extracted from research paper abstracts will be shown. " \
                  "The given two claims seem to be contradictory as they are implying" \
                  " opposite results about the same question. " \
                  "Precisely though, the two claims may have been obtained" \
                  " for different population or intervention details " \
                  "that make it possible that both claims to be true." \
                  " We want to annotate all the words that" \
                  " express opposite results."
    format_instruction = "Select all such words for each of Claim1 and Claim2. " \
                         "Print results in a json format with the key \"claim1\" and \"claim2\""
    problem = "Claim 1: {}\nClaim 2: {}"
    return instruction + "\n" + format_instruction + "\n\n" + problem


def get_log_path_chat_gpt(engine, label):
    log_path = path_join(output_path, "alamri_annotation1", "gpt", f"{engine}_req_{label}.json")
    return log_path


def get_chat_gpt_requester(engine, label) -> GPTRequester:
    template = {
        'mismatch': get_mismatch_prediction_prompt_template_chat_gpt(),
        'conflict': get_conflict_prediction_prompt_template_chat_gpt()
    }[label]
    log_path = get_log_path_chat_gpt(engine, label)
    return GPTRequester(
        OpenAIProxy(engine),
        template,
        log_path,
    )


def parse_from_json_answer(s):
    try:
        j = json.loads(s)
    except JSONDecodeError:
        print(s)
        raise

    try:
        c1 = j['claim1']
        c2 = j['claim2']
    except KeyError:
        c1 = j['Claim1']
        c2 = j['Claim2']

    def reform(c):
        if type(c) == str:
            return [c]
        else:
            return c
    return reform(c1), reform(c2)


def get_chat_gpt_file_solver(engine, label) -> GPTSolverFileRead:
    log_path = get_log_path_chat_gpt(engine, label)
    j_d = load_json_log(log_path)
    parse_answer = parse_from_json_answer
    return GPTSolverFileRead(
        j_d,
        parse_chat_gpt_response,
        parse_answer,
        get_score_from_answer_spans_chat
    )

