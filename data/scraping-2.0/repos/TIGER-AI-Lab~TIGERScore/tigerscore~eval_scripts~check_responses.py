"""
    This script is used to generate the check prompts for the responses of the candidates.
    We use GPT-4 to check the responses of the candidates.
"""
import sys
sys.path.append("..")
from xgptscore.openai_utils import openai_completions, _chatml_to_prompt
import fire
import json
import logging
import random
from string import Template

logging.basicConfig(level=logging.INFO)

# check_template = """
# Instruction: ${instruction}
# Source: ${source}
# Model-generated output: ${model_generated}
# Identified errors in the model-generated output:
# ${identified_errors}

# For each error, answer the following questions:
# Q1: Extracted pointed error phrase and the correction phrase in the explantion text. The error phrase is a substr in the model-generated output and the correction phrase is the correction of it in the explnation. The Fill null in the answer if error phrase or correction phrase is not found. (Answer [str, str]).
# A1:
# Q2: Does this error indeed occur in the model-generated output? An error might be (Answer Yes/No)
# A2:
# Q3: If A1 contains both the error phrase and the correction phrase, is the correction actually fixing the error considering the instruction and source input? (Answer Yes/No)
# A3:
# Q4: Is the reduction score appropriate considering the severity and the sentence context? (Answer Yes/No)
# A4:
# Q5: Does the explanation provide a helpful content for understanding the error and how to fix it? (Answer Yes/No)
# A5:
# Q6: Does the explanation contain some hallucinated content outside the provided instruction, source input, and model-generated output? (Answer Yes/No)
# A6:
# The output format will be in JSON
# {
# error_1: {Q1: A1, Q2: A2, Q3: A3, Q4: A4, Q5: A5, Q6: A6},
# ...
# }
# """

check_template = """
Instruction: ${instruction}
Source: ${source}
Model-generated output: ${model_generated}
Identified errors in the model-generated output:
${identified_errors}

For each error, answer the following questions:
Q1: Extracted pointed error phrase and the correction phrase in the explantion text. The error phrase is a substr in the model-generated output and the correction phrase is the correction of it in the explnation. The Fill null in the answer if error phrase or correction phrase is not found. (Answer [str, str]).
A1:
Q2: Does this error indeed occur in the model-generated output? An error might be (Answer Yes/No)
A2:
Q3: If A1 contains both the error phrase and the correction phrase, is the correction actually fixing the error considering the instruction and source input? (Answer Yes/No)
A3:
Q4: Is the reduction score appropriate considering the severity and the sentence context? (Answer Yes/No)
A4:
Q5: Does the explanation provide a helpful content for understanding the error and how to fix it? (Answer Yes/No)
A5:
Q6: Does the explanation contain some hallucinated content outside the provided instruction, source input, and model-generated output? (Answer Yes/No)
A6:
The output format will be in JSON
{
error_1: {Q1: A1, Q2: A2, Q3: A3, Q4: A4, Q5: A5, Q6: A6},
...
}
"""

# check_template = """
# Instruction: ${instruction}
# Source: ${source}
# Model-generated output: ${model_generated}
# Identified errors along with their explanations in the model-generated output are as follows:
# ${identified_errors}

# However, some of the identified errors might contain some problems. Some problems types includes:
# - The pointed error location does not contain any error
# - The pointed error location contains an error, but the correction is actually same as the model-generated output.
# - The pointed error location contains an error, but the correction does not fix the error.

# For each identified error, considering the instruction and source text, answer whether the identified error contains any problem mentioned above.
# Output choices are as follows:
# - true: for the identified error contains any problem mentioned above
# - false: for the identified error does not contain any problem mentioned above. That is, the identified error is indeed an error; the explanation is correct and makes sense; the correction does fix the error
# Output format:
# {"error_1": true/false, ...}
# """

check_template_for_no_errors = """
Instruction: ${instruction}
Source: ${source}
Model-generated output: ${model_generated}
An annotator has been asked to check if there is any error in the model-generated output. The annotator has answered that there is no error in the model-generated output.
However, the annotator might miss some errors. Please check if there is any error in the model-generated output.
Output format:
T/F (T: true for there is an error in the model-generated output; F: false for there is no error in the model-generated output)
"""


def get_check_prompts(
    item: dict
):
    check_prompts = []
    for cand in item['candidates']:
        if not isinstance(cand['responses'][1], dict):
            check_prompts.append(None)
            continue
        if len(cand['responses'][1]['errors']) == 0:
            check_prompt = Template(check_template_for_no_errors).substitute(
                instruction=item['instruction'],
                source=item['input'],
                model_generated=cand['text'],
            )
        else:
            # shuffle errors
            shuffled_errors = list(cand['responses'][1]['errors'].values())
            random.shuffle(shuffled_errors)
            cand['responses'][1]["errors"] = {
                f"error_{i+1}": v for i, v in enumerate(shuffled_errors)}
            check_prompt = Template(check_template).substitute(
                instruction=item['instruction'],
                source=item['input'],
                reference=item.get('output') or item.get("refs")[0],
                model_generated=cand['text'],
                identified_errors=json.dumps(
                    cand['responses'][1], ensure_ascii=False, indent=0),
            )
        check_prompt = check_prompt.strip(" \n\t")
        check_prompts.append(check_prompt)

    return check_prompts


def main(
    input_file: str,
    output_file: str,
    seed: int = 42,
    model_name: str = "ChatGPT",
):
    random.seed(seed)
    with open(input_file, "r") as f:
        data = json.load(f)[:10]
    check_prompts = list(map(get_check_prompts, data))
    full_check_prompts = [
        item for sublist in check_prompts for item in sublist]
    check_prompts = [x for x in full_check_prompts if x is not None]
    print(len(check_prompts))
    print(len(full_check_prompts))
    chatmls = [[{"role": "system", "content": "You are an AI assistant that helps people find information."},
                {"role": "user", "content": prompt}] for prompt in check_prompts]
    chatml_prompts = [_chatml_to_prompt(chatml) for chatml in chatmls]

    decoding_kwargs = {
        # "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1.0,
        "timeout": 30,
        "request_timeout": 30
    }
    results = openai_completions(
        chatml_prompts, model_name=model_name, **decoding_kwargs)
    logging.info("Total price: {:.4f}$".format(
        sum(results['price_per_example'])))
    completions = results['completions']

    def loads_json(json_str: str):
        try:
            return json.loads(json_str)
        except Exception:
            return json_str
    idx = 0
    full_idx = 0
    for item in data:
        for cand in item['candidates']:
            if full_check_prompts[idx] is None:
                cand['check_prompt'] = None
                cand['check_response'] = None
            else:
                cand['check_prompt'] = check_prompts[idx]
                cand['check_response'] = loads_json(completions[idx])
                idx += 1
            full_idx += 1
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved to {output_file}")

    # # counter the distribution of each QA
    # qa_answers = {f"Q{i+1}" : [] for i in range(11)}
    # for item in data:
    #     for cand in item['candidates']:
    #         if isinstance(cand['check_response'], dict):
    #             for q in cand['check_response']:
    #                 if re.match(r"Q\d+", q):
    #                     qa_answers[q].append(cand['check_response'][q])
    #                 elif re.match(r"error_\d+", q):
    #                     for q2 in cand['check_response'][q]:
    #                         if re.match(r"Q\d+", q2):
    #                             qa_answers[q2].append(cand['check_response'][q][q2])
    #                         else:
    #                             raise ValueError(f"Invalid question {q2}")
    #                 else:
    #                     raise ValueError(f"Invalid question {q}")
    # for q in qa_answers:
    #     if q == "Q1":
    #         logging.info(f"{q}: {Counter([type(item) for item in qa_answers[q]])}")
    #     else:
    #         logging.info(f"{q}: {Counter([str(item) for item in qa_answers[q]])}")

    # # filter out the canidates based on the answers
    # filtered_data = copy.deepcopy(data)
    # for item in filtered_data:
    #     for i, cand in enumerate(item['candidates']):
    #         flag = True
    #         keep_cand_idxs = []
    #         if not isinstance(cand['check_response'], dict):
    #             continue
    #         for key in cand['check_response']:
    #             if re.match(r'error_\d+', key):
    #                 for sub_key in cand['check_response'][key]:
    #                     if sub_key == "Q1":
    #                         pass
    #                     elif sub_key == "Q2":
    #                         if cand['check_response'][key][sub_key] == "No":
    #                             flag = False
    #                             break
    #                     elif sub_key == "Q3":
    #                         if cand['check_response'][key][sub_key] == "No":
    #                             flag = False
    #                             break
    #                     elif sub_key == "Q4":
    #                         if isinstance(cand['check_response'][key][sub_key], int):
    #                             cand['responses'][-1]['errors'][key]['reduction_score'] = cand['check_response'][key][sub_key]
    #                     elif sub_key == "Q5":
    #                         if cand['check_response'][key][sub_key] == "No":
    #                             flag = False
    #                             break
    #                     elif sub_key == "Q5":
    #                         if cand['check_response'][key][sub_key] == "No":
    #                             flag = False
    #                             break
    #                     elif sub_key == "Q6":
    #                         if cand['check_response'][key][sub_key] == "Yes":
    #                             flag = False
    #                             break
    #                     else:
    #                         raise ValueError(f"Invalid question {sub_key}")
    #         if flag:
    #             keep_cand_idxs.append(i)
    #     item['candidates'] = [item['candidates'][i] for i in keep_cand_idxs]
    # # count the number of removed candidates
    # num_removed = 0
    # ori_total_num_cands = 0
    # for filtered_item, item in zip(filtered_data, data):
    #     num_removed += len(item['candidates']) - len(filtered_item['candidates'])
    #     ori_total_num_cands += len(item['candidates'])
    # logging.info(f"Removed {num_removed} candidates from {ori_total_num_cands} candidates")
    # # filter out the items with no candidates
    # filtered_data = [item for item in filtered_data if len(item['candidates']) > 0]
    # logging.info(f"Filtered {len(data) - len(filtered_data)} items")
    # # save the filtered data
    # with open(output_file.replace(".json", "_filtered.json"), "w") as f:
    #     json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    # logging.info(f"Saved to {output_file.replace('.json', '_filtered.json')}")


if __name__ == "__main__":
    fire.Fire(main)
