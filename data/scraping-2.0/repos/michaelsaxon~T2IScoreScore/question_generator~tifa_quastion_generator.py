import openai
from openai_utils import openai_setup, openai_completion
import csv
import json
from absl import flags
import sys


FLAGS = flags.FLAGS

flags.DEFINE_string("api_key", "", "OpenAI API key.")
flags.DEFINE_string("input_file", "", "Path to the input file with prompts.")
flags.DEFINE_string("output_csv", "", "Path to save the output CSV file.")

class TIFA_QuestionGenerator:
    def __init__(self):
        with open('/content/TIFA_prompt.txt', 'r') as file:
            self.prompt = file.read()
            self.categories = ['object',
                                'human',
                                'animal',
                                'food',
                                'activity',
                                'attribute',
                                'counting',
                                'color',
                                'material',
                                'spatial',
                                'location',
                                'shape',
                                'other']

    def parse_resp(self, resp):
        resp = resp.split('\n')

        question_instances = []

        this_entity = None
        this_type = None
        this_question = None
        this_choices = None
        this_answer = None

        for line_number in range(6, len(resp)):
            line = resp[line_number]
            if line.startswith('About '):
                whole_line = line[len('About '):-1]
                this_entity = whole_line.split(' (')[0]
                this_type = whole_line.split(' (')[1].split(')')[0]

            elif line.startswith('Q: '):
                this_question = line[3:]
            elif line.startswith('Choices: '):
                this_choices = line[9:].split(', ')
            elif line.startswith('A: '):
                this_answer = line[3:]

                if this_entity and this_question and this_choices:
                    question_instances.append((this_entity, this_question, this_choices, this_answer, this_type))
                this_question = None
                this_choices = None
                this_answer = None

        return question_instances


    def get_question_and_answers(self, caption):
        this_prompt = self.prompt + caption + "\nEntities"

        resp = openai_completion(this_prompt)
        question_instances = self.parse_resp(resp)

        this_caption_qas = []

        for question_instance in question_instances:
            this_qa = {}
            this_qa['caption'] = caption
            this_qa['element'] = question_instance[0]
            this_qa['question'] = question_instance[1]
            this_qa['choices'] = question_instance[2]
            this_qa['answer'] = question_instance[3]
            this_qa['element_type'] = question_instance[4]

            if question_instance[4] not in self.categories:
                continue

            if this_qa['element_type'] in ['animal', 'human']:
                this_qa['element_type'] = 'animal/human'

            this_caption_qas.append(this_qa)

        return this_caption_qas

    def save_to_csv(self, prompts, csv_output_path):
        csv_data = {'id': [], 'prompt': [], 'question_id': [], 'question': [], 'choices': [], 'answer': []}

        for i, prompt in enumerate(prompts):
            prompt_qas = self.get_question_and_answers(prompt)
            print(prompt_qas)
            for j, question in enumerate(prompt_qas):
                csv_data['id'].append(f'{i}')
                csv_data['prompt'].append(prompt)
                csv_data['question'].append(question['question'])
                csv_data['question_id'].append(f'{j}')
                csv_data['choices'].append("|".join(question['choices']))
                csv_data['answer'].append(question['answer'])

        with open(csv_output_path, 'w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['id', 'prompt', 'question_id', 'question', 'choices', 'answer']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(csv_data['id'])):
                writer.writerow({
                    'id': csv_data['id'][i],
                    'prompt': csv_data['prompt'][i],
                    'question_id': csv_data['question_id'][i],
                    'question': csv_data['question'][i],
                    'choices': csv_data['choices'][i],
                    'answer': csv_data['answer'][i],
                })

if __name__ == "__main__":
    FLAGS(sys.argv)  # Parse command line flags.

    api_key = FLAGS.api_key
    input_file_path = FLAGS.input_file
    output_csv_path = FLAGS.output_csv

    question_generator = TIFA_QuestionGenerator()

    with open(input_file_path, 'r') as file:
        prompts = file.readlines()

    prompts = [prompt.strip() for prompt in prompts]

    question_generator.save_to_csv(prompts, output_csv_path)

