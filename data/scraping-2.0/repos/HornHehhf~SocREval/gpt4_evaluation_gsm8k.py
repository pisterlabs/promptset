import os
import openai
import time
import json
from scipy.stats import somersd

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

instruction = \
"Does the generated response answer the question in a well-justified manner? Please give me an overall quality score in [1, 2, 3, 4, 5] (1=incomprehensible and wrong, 5=clear and correct)."

example_question = \
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

example_generated_response = \
    "Janet eats 3 duck eggs for breakfast and bakes 4 into muffins so 3 + 4 = <<3+4=7>>7 duck eggs are used\nEach day Janet's ducks lay 16 eggs and she uses 7, 16 - 7 = <<16-7=9>>9 duck eggs are for sale\nShe sells her eggs for $2 per egg and has 9 available for sale so 2 * 9 = $<<2*9=18>>18 per day\nA: 18\n"

example_representation = {'overall quality': 5}

example_representation = json.dumps(example_representation)

evaluation_prompt = "Please give me the overall quality of the generated response for the question based on the instruction and the format of the example representation:"


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True


def run_gpt4_evaluation_gsm8k(test_path, output_path, input_option, output_option):
    max_num_per_question = 5
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        question = test_case['question']
        generated_response = test_case[input_option]['solution']
        combined_prompt = "Instruction:\n" + instruction \
                          + "\n\nExample question:\n" + example_question \
                          + "\n\nExample generated response:\n" + example_generated_response \
                          + "\n\nExample representation:\n" + example_representation \
                          + "\n\nQuestion:\n" + question \
                          + "\n\nGenerated response:\n" + generated_response \
                          + "\n\n" + evaluation_prompt + "\n"
        print(combined_prompt)
        for i in range(max_num_per_question):
            response = openai.ChatCompletion.create(model="gpt-4-0613",
                                                    messages=[{"role": "user", "content": combined_prompt}])
            text = response['choices'][0]['message']['content']
            if i == max_num_per_question - 1:
                test_case[output_option] = text
            if is_json(text):
                representation = json.loads(text)
                print(representation)
                test_case[output_option] = json.dumps(representation)
                break
            else:
                print('format error')
        with open(output_path, 'w') as f:
            json.dump(test_data, f, indent=4)


def evaluate_gpt4_evaluation_gsm8k(test_path, representation_option):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    correctness_list = []
    human_eval_score_list = []
    for idx, test_case in enumerate(test_data):
        representation = json.loads(test_case[representation_option])
        correctness = (float(representation['overall quality']) - 1) / 4
        human_eval_score = (float(test_case['0_full_newOverall_result']) - 1) / 4
        human_eval_score_list.append(human_eval_score)
        correctness_list.append(correctness)
    print('\ncorrectness somersd:', somersd(human_eval_score_list, correctness_list))


if __name__ == '__main__':
    dir_path = '/path/to/working/dir/'
    gsm8k_roscoe_path = dir_path + 'ROSCOE/gsm8k_roscoe_annotations.json'
    gsm8k_roscoe_gpt4_evaluation_path = dir_path + 'ROSCOE/gsm8k_roscoe_annotations_gpt4_evaluation.json'
    time_start = time.time()
    run_gpt4_evaluation_gsm8k(gsm8k_roscoe_path, gsm8k_roscoe_gpt4_evaluation_path, input_option='175b_verification',
                              output_option='175b_verification_gpt4_evaluation')
    evaluate_gpt4_evaluation_gsm8k(gsm8k_roscoe_gpt4_evaluation_path,
                                   representation_option='175b_verification_gpt4_evaluation')
    time_end = time.time()
    print('\ntime:', time_end - time_start)

