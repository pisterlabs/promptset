from contradiction.medical_claims.token_tagging.gpt_solver.gpt_solver import GPTSolver, GPTRequester, GPTSolverFileRead, \
    get_parse_answer_texts_for_instruct, load_json_log, get_score_from_answer_spans
from cpath import output_path
from misc_lib import path_join
from utils.open_ai_api import OpenAIProxy, parse_instruct_gpt_response


def get_mismatch_prediction_prompt_template():
    instruction = "In each of the examples, " \
                  "two claims extracted from research paper abstracts will be shown. " \
                  "The given two claims seem to be contradictory as they are implying" \
                  " opposite results about the same question. " \
                  "Precisely though, the two claims may have been obtained" \
                  " for different population or intervention details " \
                  "that make it possible that both claims to be true." \
                  " We want to annotate the tokens (words) that" \
                  " express different conditions."

    problem = "Claim 1: {}\nClaim 2: {}"
    later_template = "Condition tokens in Claim 1:"
    return instruction + "\n\n" + problem + "\n\n" + later_template


def get_conflict_prediction_prompt_template():
    instruction = "In each of the examples, " \
                  "two claims extracted from research paper abstracts will be shown. " \
                  "The given two claims seem to be contradictory as they are implying" \
                  " opposite results about the same question. " \
                  "Precisely though, the two claims may have been obtained" \
                  " for different population or intervention details " \
                  "that make it possible that both claims to be true." \
                  " We want to annotate the tokens (words) that" \
                  " express opposite results."

    problem = "Claim 1: {}\nClaim 2: {}"
    later_template = "Opposite results tokens in Claim 1:"
    return instruction + "\n\n" + problem + "\n\n" + later_template



def get_gpt_solver_mismatch() -> GPTSolver:
    log_path = path_join(output_path, "alamri_annotation1", "gpt", "davinci_mismatch.json")
    template = get_mismatch_prediction_prompt_template()
    claim2_pattern = "Condition tokens in Claim 2:"
    parse_answer = get_parse_answer_texts_for_instruct(
        template,
        claim2_pattern)

    return GPTSolver(
        OpenAIProxy("text-davinci-003"),
        template,
        "Condition tokens in Claim 2:",
        log_path,
        parse_instruct_gpt_response,
        parse_answer
    )


def get_gpt_requester_mismatch() -> GPTRequester:
    log_path = path_join(output_path, "alamri_annotation1", "gpt", "davinci_req_mismatch.json")
    template = get_mismatch_prediction_prompt_template()
    return GPTRequester(
        OpenAIProxy("text-davinci-003"),
        template,
        log_path,
    )


def get_gpt_requester_conflict() -> GPTRequester:
    log_path = path_join(output_path, "alamri_annotation1", "gpt", "davinci_req_conflict.json")
    template = get_conflict_prediction_prompt_template()
    return GPTRequester(
        OpenAIProxy("text-davinci-003"),
        template,
        log_path)


def get_gpt_file_solver_instruct_common(claim2_pattern, log_path, template):
    j_d = load_json_log(log_path)
    parse_answer = get_parse_answer_texts_for_instruct(
        template,
        claim2_pattern)
    return GPTSolverFileRead(
        j_d,
        parse_instruct_gpt_response,
        parse_answer,
        get_score_from_answer_spans
    )


def get_gpt_file_solver_mismatch() -> GPTSolverFileRead:
    template = get_mismatch_prediction_prompt_template()
    claim2_pattern = "Condition tokens in Claim 2:"
    log_path = path_join(output_path, "alamri_annotation1", "gpt", "davinci_req_mismatch.mod.json")
    return get_gpt_file_solver_instruct_common(claim2_pattern, log_path, template)


def get_gpt_file_solver_conflict() -> GPTSolverFileRead:
    template = get_conflict_prediction_prompt_template()
    claim2_pattern = "Opposite results tokens in Claim 2"
    log_path = path_join(output_path, "alamri_annotation1", "gpt", "davinci_req_conflict.mod.json")
    return get_gpt_file_solver_instruct_common(claim2_pattern, log_path, template)

