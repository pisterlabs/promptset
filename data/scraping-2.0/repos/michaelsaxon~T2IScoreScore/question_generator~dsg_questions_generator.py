import json
import pandas as pd
from absl import flags
import sys
import subprocess
import openai
from openai_utils import openai_setup, openai_completion
from query_utils import generate_dsg


FLAGS = flags.FLAGS

flags.DEFINE_string("api_key", "", "OpenAI API key.")
flags.DEFINE_string("input_file", "", "Path to the input file with prompts.")
flags.DEFINE_string("output_csv", "", "Path to save the output CSV file.")

class DSG_QuestionGenerator:
    def __init__(self, api_key):
        self.api_key = api_key

    def process_prompts(self, input_file_path, csv_output_path):
        prompts = self.read_prompts_from_file(input_file_path)

        csv_data = {'id': [], 'prompt': [], 'question_id': [], 'question': [], 'choices': [], 'answer': []}

        for i, prompt in enumerate(prompts):
            INPUT_TEXT_PROMPT = prompt
            id2prompts = {
                f'{i}': {
                    'input': INPUT_TEXT_PROMPT,
                }
            }

            id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
                id2prompts,
                generate_fn=openai_completion)

              # Extract and store the questions in a list
            questions = id2question_outputs[f'{i}']['output'].split('\n')
            questions = [q.strip().split('|')[-1].strip() for q in questions if q.strip()]  # Extract the question part

            for j, question in enumerate(questions):
                csv_data['id'].append(f'{i}')
                csv_data['prompt'].append(prompt)
                csv_data['question'].append(question)
                csv_data['question_id'].append(f'{j}')
                csv_data['choices'].append("|".join(["yes", "no"]))
                csv_data['answer'].append("yes")

        df = pd.DataFrame(csv_data)
        df.to_csv(csv_output_path, index=False)

    def read_prompts_from_file(self, input_file_path):
        with open(input_file_path, 'r') as file:
            prompts = file.readlines()
        return [prompt.strip() for prompt in prompts]


if __name__ == "__main__":
    FLAGS(sys.argv)  # Parse command line flags.

    api_key = FLAGS.api_key
    input_file_path = FLAGS.input_file
    output_csv_path = FLAGS.output_csv

    question_creator = DSG_QuestionGenerator(api_key)
    repository_url = "https://github.com/j-min/DSG.git"
    target_directory = "SGD"

    subprocess.run(["git", "clone", repository_url, target_directory])
    subprocess.run(["cd", target_directory])

    question_creator.process_prompts(input_file_path, output_csv_path)
