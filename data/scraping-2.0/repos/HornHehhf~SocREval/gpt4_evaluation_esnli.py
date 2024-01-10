import os
import openai
import time
import json
from scipy.stats import somersd


# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

instruction = \
"Does the generated response answer the question in a well-justified manner? Please give me an overall quality score in [1, 2, 3, 4, 5] (1=incomprehensible and wrong, 5=clear and correct)."

example_premise = "Two women are embracing while holding to go packages."

example_hypothesis = "Two woman are holding packages."

generic_question = "Is the Claim supported by the Situation?"

example_generated_response = "The two women are most likely embracing because they are either friends or family. If they were just holding packages, there would be no need for them to embrace. The answer is Yes."

example_representation = {'overall quality': 4}

example_representation = json.dumps(example_representation)

evaluation_prompt = "Please give me the overall quality of the generated response for the question based on the instruction and the format of the example representation:"


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True


def run_gpt4_evaluation_esnli(test_path, output_path, input_option, output_option):
    max_num_per_question = 5
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        premise = test_case["premise"]
        hypothesis = test_case["hypothesis"]
        generated_response = test_case[input_option]
        combined_prompt = "Instruction:\n" + instruction \
                          + "\n\nExample Situation (Premise):\n" + example_premise \
                          + "\n\nExample Claim (Hypothesis):\n" + example_hypothesis \
                          + "\n\nExample question:\n" + generic_question \
                          + "\n\nExample generated response:\n" + example_generated_response \
                          + "\n\nExample representation:\n" + example_representation \
                          + "\n\nSituation (Premise):\n" + premise \
                          + "\n\nClaim (Hypothesis):\n" + hypothesis \
                          + "\n\nQuestion:\n" + generic_question \
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


def evaluate_gpt4_evaluation_esnli(test_path, representation_option):
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
    esnli_roscoe_path = dir_path + 'ROSCOE/esnli_roscoe_annotations.json'
    esnli_roscoe_gpt4_evaluation_path = dir_path + 'ROSCOE/esnli_roscoe_annotations_gpt4_evaluation.json'
    time_start = time.time()
    run_gpt4_evaluation_esnli(esnli_roscoe_path, esnli_roscoe_gpt4_evaluation_path,
                              input_option='gpt-3', output_option='gpt-3_gpt4_evaluation')
    evaluate_gpt4_evaluation_esnli(esnli_roscoe_gpt4_evaluation_path,
                                   representation_option='gpt-3_gpt4_evaluation')
    time_end = time.time()
    print('\ntime:', time_end - time_start)

