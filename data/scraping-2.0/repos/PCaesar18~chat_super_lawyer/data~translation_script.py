

import openai
import pandas as pd
from datasets import load_dataset

# Set OpenAI API key

openai.api_key = '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'


# Define a function for streaming the dataset
def stream_dataset(dataset_name="ehartford/dolphin", split='train', num_samples=1):
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    prompts = []
    for idx, item in enumerate(dataset):
        # extract the instruction, input, and output
        instruction = item['instruction']
        inputstr = item.get('input', None)  # use get in case the key 'input' doesn't exist
        outputstr = item['output']

        # create translation prompt and print
        prompt = create_translation_prompt(instruction, inputstr, outputstr)
        prompts.append(prompt)

        if idx + 1 == num_samples:
            break

    return prompts


def create_translation_prompt(instruction, inputstr, outputstr, src_lang="English", tgt_lang="Dutch"):
    TRANSLATION_PROMPT = f"""You are asked to translate a task's instruction, optional input to the task, and the output of the task, from {src_lang} into {tgt_lang}.

    Here are the requirements that you should adhere to:
    1. maintain the format: the task consists of a task instruction (marked `instruction: `), optional input to the task (marked `input: `) and output for the task marked with `output: `;
    2. do not translate the identifiers `instruction: `, `input: `, and `output: ` but instead copy them to your output;
    3. make sure that text is fluent to read and does not contain grammatical errors. Use standard {tgt_lang} without regional bias;
    4. translate the instruction and input text using informal, but standard, language;
    5. if the instruction is to correct grammar mistakes or spelling mistakes then you have to generate a similar mistake in the input in {tgt_lang}, and then also generate a corrected output version in the output in {tgt_lang};
    6. if the instruction is to translate text from one language to another, then you do not translate the text that needs to be translated in the instruction or the input, nor the translation in the output (just copy them as-is);
    7. do not translate code fragments but copy them to your output. If there are English examples, variable names or definitions in code fragments, keep them in English.

    Now translate the following task with the requirements set out above. Do not provide an explanation and do not add anything else.\n\n"""

    task_text = f'instruction: "{instruction}"\n\n'
    if inputstr:
        task_text += f'input: "{inputstr}"\n\n'
    task_text += f'output: "{outputstr}"'

    return TRANSLATION_PROMPT + task_text


#
# Function to translate text
def translate(prompt):



    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates English to Dutch to the requirements that are given to you."},
            {"role": "user", "content": prompt},
        ],
    )

    return response['choices'][0]['message']['content'].strip()

test = stream_dataset()
translate_prompt = translate(test[0])
print(translate_prompt)


