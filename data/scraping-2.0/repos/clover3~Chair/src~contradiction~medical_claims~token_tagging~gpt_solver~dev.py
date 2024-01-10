import json

from contradiction.medical_claims.token_tagging.gpt_solver.gpt_solver import get_score_from_j
from contradiction.medical_claims.token_tagging.gpt_solver.get_instruct_gpt_solver import get_mismatch_prediction_prompt_template
from utils.open_ai_api import OpenAIProxy
from cpath import output_path
from misc_lib import path_join


def main():
    claim1 = "Supplementation during pregnancy with a medical food containing L-arginine and antioxidant vitamins reduced the incidence of pre-eclampsia in a population at high risk of the condition."
    claim2 = "Oral L-arginine supplementation did not reduce mean diastolic blood pressure after 2 days of treatment compared with placebo in pre-eclamptic patients with gestational length varying from 28 to 36 weeks."
    tokens1 = claim1.split()
    tokens2 = claim2.split()

    prompt = get_mismatch_prediction_prompt_template().format(claim1, claim2)
    # print("prompt: ", prompt)
    # proxy = OpenAIProxy("text-davinci-003")
    # j_output = proxy.request(prompt)
    # print(j_output)
    # open(path_join(output_path, "alamri_annotation1", "gpt", "msg.json"), "w").write(json.dumps(j_output))
    #
    j_output_s: str = open(path_join(output_path, "alamri_annotation1", "gpt", "msg.json"), "r").read()
    j_output = json.loads(j_output_s)
    print(j_output)
    claim2_pattern = "Condition tokens in Claim 2:"
    spair = get_score_from_j(prompt, tokens1, tokens2, j_output, claim2_pattern)
    print(spair)


if __name__ == "__main__":
    main()