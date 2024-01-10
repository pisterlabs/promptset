import argparse
import os
import json

import openai


parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input", help="input task file or dir containing task files")
parser.add_argument('-o', "--output", default="gpt3-augment-results", help="output dir")
parser.add_argument('-t', "--template", default="choice_expl.prompt", help="template file")

# GPT-3 generation hyperparameters
parser.add_argument('--engine', type=str, default='text-davinci-002',
                    choices=['ada',
                             'text-ada-001',
                             'babbage',
                             'text-babbage-001',
                             'curie',
                             'text-curie-001',
                             'davinci',
                             'text-davinci-001',
                             'text-davinci-002'],
                    help='The GPT-3 engine to use.')  # choices are from the smallest to the largest model
parser.add_argument('--max_tokens', type=int, default=120, required=False, help='')
parser.add_argument('--temperature', type=float, default=0.7, required=False, help='')
parser.add_argument('--top_p', type=float, default=1, required=False, help='')
parser.add_argument('--frequency_penalty', type=float, default=0.0, required=False, help='')
parser.add_argument('--presence_penalty', type=float, default=0.0, required=False, help='')
parser.add_argument('--stop_tokens', nargs='+', type=str,
                        default=None, required=False, help='Stop tokens for generation')
args = parser.parse_args()

def fill_template(prompt_template_file: str, **prompt_parameter_values) -> str:
    prompt = ''
    with open(prompt_template_file) as prompt_template_file:
        for line in prompt_template_file:
            if line.startswith('#'):
                continue  # ignore comment lines in the template
            prompt += line
    for parameter, value in prompt_parameter_values.items():
        prompt = prompt.replace('{'+parameter+'}', value)
    return prompt

def generate_one(input_text: str, args) -> str:
    generation_output = openai.Completion.create(engine=args.engine,
                                                 prompt=input_text,
                                                 max_tokens=max(args.max_tokens, len(input_text.split(' '))),
                                                 temperature=args.temperature,
                                                 top_p=args.top_p,
                                                 frequency_penalty=args.frequency_penalty,
                                                 presence_penalty=args.presence_penalty,
                                                 best_of=1,
                                                 stop=args.stop_tokens,
                                                 logprobs=0,  # log probability of top tokens
                                                 )
    generation_output = generation_output['choices'][0]['text']
    generation_output = generation_output.strip()
    return generation_output

if not os.path.exists(args.output):
    os.makedirs(args.output)

task_paths = []
if os.path.isdir(args.input):
    task_paths = [os.path.join(args.input, filename) for filename in os.listdir(args.input)]
else:
    task_paths = [args.input]

task_names = []
orig_prompts = [] # the Definition in the original task
prompts = [] # templated prompt for gpt3 consisting of formatted pos examples and possible choices
results = []
for task_file in task_paths:
    task_dict = json.load(open(task_file, 'r'))
    instruction = "\n".join(task_dict['Definition'])
    if not instruction.endswith('.'):
        instruction += '.' # prevent gpt3 generating a dot

    pos_examples = task_dict["Positive Examples"]
    pos_examples_by_class = {}
    for pos_example in pos_examples:
        example_output = pos_example["output"]
        if example_output in pos_examples_by_class:
            pos_examples_by_class[example_output].append(pos_example)
        else:
            pos_examples_by_class[example_output] = [pos_example]

    examples_prompt = ''
    num_examples = 1
    for cls, examples in pos_examples_by_class.items():
        first_example = examples[0] # select 1 example per class
        examples_prompt += f"({num_examples})\n"
        example_input = first_example["input"]
        examples_prompt += "Input: " + example_input + "\n"
        example_output = first_example["output"]
        examples_prompt += "Output: " + example_output + "\n"
        example_explanation = first_example["explanation"]
        examples_prompt += "Explanation: " + example_explanation + "\n"
        num_examples += 1
    classes_prompt = ""
    for cls in pos_examples_by_class.keys():
        classes_prompt += f"'{cls}', "
    classes_prompt = classes_prompt.strip(", ")
    
    orig_prompts.append(instruction)
    task_name = os.path.basename(task_file)
    task_names.append(task_name)
    prompt = fill_template(args.template, defintion=instruction, examples=examples_prompt, classes=classes_prompt)
    prompts.append(prompt)
    result = generate_one(prompt, args)
    result = result.split("one by one:")[-1].replace("\n", " ")
    results.append(result)


for task_name, orig_prompt, gen_prompts in zip(task_names, orig_prompts, results):
    augmented_prompt = orig_prompt + " " + gen_prompts # concat original prompt and generation class defs 
    data_dict = {
        'orignal_task': task_name,
        'action': "choice explanation",
        'original_prompt': orig_prompt,
        'generated_prompts': [augmented_prompt]
    }
    save_file = os.path.join(args.output, args.template + '_' +task_name)
    json.dump(data_dict, open(save_file, 'w'), indent=4)