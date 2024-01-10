import os
import openai

# os.environ['TRANSFORMERS_CACHE'] = '/network/scratch/m/megh.thakkar/huggingface/models'
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from core import filter_code, run_eval, fix_indents
import os
import torch
import json
from openai import OpenAI
import dotenv
dotenv.load_dotenv()
import time


# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = "" # token for huggingface

# SIMPLE PROMPT ZERO SHOT + STEPS
def construct_simple_prompt(problem,thought):
    PROMPT_M = f'''You are great at python programming. You shall be provided with a programming Task and a Solving process to solve the task. 
You need to complete the programming task by thinking step by step. 

Solving process:
{thought}

Programming Task:
{problem}
'''
    return PROMPT_M


def load_json(file_path = "./humaneval_steps_magicoder.json"):
    with open(file_path, "r") as json_file:
        data_dict = json.load(json_file)
    
    list_prompts = []
    for k,v in data_dict.items():
        list_prompts.append(v)
    # sorted_dict = {int(key): value for key, value in sorted(data_dict.items())}
    # sorted_value_list = [value for key, value in sorted_dict.items()]
    return list_prompts


# ZERO SHOT PROMPT  (WITHOUT STEPS)
def wrap_code_template(instruction):
    P = f'''
You must complete the python function I give you. When doing so, you must write the completion in the following form:
${{ORIG_FUNCTION}}
    <|start_of_completion|>
    ${{INSERT_COMPLETION}}

Be sure to use the same indentation I specified. Furthermore, you may only write your response in code.

ORIG_FUNCTION=
{instruction}

Once more, please follow the template by repeating the original function, including the <|start_of_completion|> token, then writing the completion.
'''
    return P

# ONESHOT PROMPT WITH STEPS/SOLVING PROCESS
def wrap_code_steps(instruction,example):
    P2 = f'''
You must complete the python function I give you. When doing so, you must write the completion in the following form:
${{ORIGINAL_FUNCTION}}
    <|start_of_completion|>
    ${{INSERT_COMPLETION}}

Be sure to use the same indentation I specified. Furthermore, you may only write your response in code.

ORIGINAL_FUNCTION=
def binary_sum(a,b):
    """
    Given two binary strings a and b, return their sum as a binary string.
    """

SOLVING PROCESS=
1. Reverse the input binary strings 'a' and 'b' to prepare for right-to-left addition.
2. Iterate through both strings simultaneously, calculating the sum of current bits along with the carry.
3. Store the resulting sum bit in 'result' and update the carry.
4. fter the loop, check if there's any remaining carry and append it to the 'result'. Finally, return the 'result' as the binary sum.

RESPONSE=
def binary_sum(a, b):
    """
    Given two binary strings a and b, return their sum as a binary string.
    """
    <|start_of_completion|>
    result = ""
    carry = 0

    # Reverse the input strings
    a = a[::-1]
    b = b[::-1]

    # Loop through the strings
    for i in range(max(len(a), len(b))):
        bit_a = int(a[i]) if i < len(a) else 0
        bit_b = int(b[i]) if i < len(b) else 0
        current_sum = bit_a + bit_b + carry
        result = str(current_sum % 2) + result
        carry = current_sum // 2

    # Check for any remaining carry
    if carry > 0:
        result = str(carry) + result

    return result

ORIGINAL_FUNCTION=
{instruction}

SOLVING PROCESS=
{example}

Once more, please follow the template by repeating the original function, including the <|start_of_completion|> token, then writing the completion.
RESPONSE='''
    P = f'''
You must complete the python function I give you. When doing so, you must write the completion in the following form:
${{ORIGINAL_FUNCTION}}
    <|start_of_completion|>
    ${{INSERT_COMPLETION}}
I am providing you an example of how to do this. Please follow the same format.

-- EXAMPLE STARTS HERE --
ORIGINAL_FUNCTION=
def binary_sum(a,b):
    """
    Given two binary strings a and b, return their sum as a binary string.
    """

SOLVING PROCESS=
1. Reverse the input binary strings 'a' and 'b' to prepare for right-to-left addition.
2. Iterate through both strings simultaneously, calculating the sum of current bits along with the carry.
3. Store the resulting sum bit in 'result' and update the carry.
4. fter the loop, check if there's any remaining carry and append it to the 'result'. Finally, return the 'result' as the binary sum.

COMPLETED FUNCTION=
def binary_sum(a, b):
    """
    Given two binary strings a and b, return their sum as a binary string.
    """
    <|start_of_completion|>
    result = ""
    carry = 0

    # Reverse the input strings
    a = a[::-1]
    b = b[::-1]

    # Loop through the strings
    for i in range(max(len(a), len(b))):
        bit_a = int(a[i]) if i < len(a) else 0
        bit_b = int(b[i]) if i < len(b) else 0
        current_sum = bit_a + bit_b + carry
        result = str(current_sum % 2) + result
        carry = current_sum // 2

    # Check for any remaining carry
    if carry > 0:
        result = str(carry) + result

    return result
-- EXAMPLE ENDS HERE --
    
You must complete the python function I give you. When doing so, you must write the completion in the following form:
${{ORIGINAL_FUNCTION}}
    <|start_of_completion|>
    ${{INSERT_COMPLETION}}

Be sure to use the same indentation I specified. Furthermore, you may only write your response in actual code.

ORIGINAL_FUNCTION=
{instruction}

SOLVING PROCESS=
{example}

Tip: Once more, please follow the template by repeating the original function, including the <|start_of_completion|> token, then writing the completion. You can only respond with actual code.
COMPLETED FUNCTION=
'''
    return P

# SIMPLE PROMPT ZERO SHOT + STEPS (v0)
def wrap_with_examples(problem, thought):
    P = f'''
Thinking hint:
{thought}
    
{problem}
'''
    return P


# ONE SHOT PROMPT WITH PSEUDOCODE VARIANT
def wrap_one_shot_pseudocode(instruction,example):
    PROMPT = f'''
You must complete the python function I give you. When doing so, you must write the completion in the following form:
${{ORIG_FUNCTION}}
    ${{INSERT_COMPLETION}}

BEGIN EXAMPLE
ORIG_FUNCTION=
You are an expert Python programmer, and you need to respond with code to complete is your task: Given two binary strings a and b, return their sum as a binary string.
Your code should pass these tests:
assert binary_sum("11", "1") == "100"

    
PSEUDOCODE=
function binary_sum with inputs a and b:
    initialize result as an empty string.
    set carry to 0.

    reverse both a and b for processing from least significant bit.

    loop over each bit position in the longer of a and b:
        take the current bit from a and b, or 0 if the string is shorter.
        sum the current bits and carry.
        add the least significant bit of the sum to the front of the result.
        update carry with the most significant bit of the sum.

    if there is a carry left, add it to the front of the result.

    return the result.
end of function

RESPONSE=
def binary_sum(a, b):
    result = ""
    carry = 0

    # Reverse the input strings
    a = a[::-1]
    b = b[::-1]

    # Loop through the strings
    for i in range(max(len(a), len(b))):
        bit_a = int(a[i]) if i < len(a) else 0
        bit_b = int(b[i]) if i < len(b) else 0
        current_sum = bit_a + bit_b + carry
        result = str(current_sum % 2) + result
        carry = current_sum // 2

    # Check for any remaining carry
    if carry > 0:
        result = str(carry) + result

    return result
END EXAMPLE

ORIG_FUNCTION=
{instruction}

PSEUDOCODE=
{example}

RESPONSE=
''' # v1 --this is the current version--
    return PROMPT

# ONE SHOT PROMPT WITH PSEUDOCODE 
def wrap_zeroshot_psuedocode(problem, thought):
    P = f'''
You must complete the python function I give you. When doing so, you must write the completion in the following form:
${{ORIG_FUNCTION}}
    <|start_of_completion|>
    ${{INSERT_COMPLETION}}

Be sure to use the same indentation I specified. Furthermore, you may only write your response in code.
You have been also provided a pseudocode to help you get started.

ORIG_FUNCTION=
{problem}

PSEUDOCODE=
{thought}

Once more, please follow the template by repeating the original function, including the <|start_of_completion|> token, then writing the completion.
'''

    P2 = f'''
You must complete the python function I give you. When doing so, you must write the completion in the following form:
${{ORIG_FUNCTION}}
    <|start_of_completion|>
    ${{INSERT_COMPLETION}}

You should use the psuedocode provided (PSUEDOCODE) to guide your reasoning process.    
Be sure to use the same indentation I specified. Furthermore, you may only write your response in code.


ORIG_FUNCTION=
{problem}

PSEUDOCODE=
{thought}

Once more, please follow the template by repeating the original function, including the <|start_of_completion|> token, then writing the completion.
'''
    return P2


def answer_query_gpt(sentence,example):
    """Answers a query based on the prompt
    Args:
        sentence (str) : Prompt to answer
    Returns:
        label (str) : Answer to the query
    """
    # s = f"{wrap_code_template(sentence)}" # zero-shot
    # s = f"{wrap_with_examples(sentence,example)}" # zero-shot + steps (simple)
    # s = f"{wrap_code_steps(sentence,example)}" # zero-shot + steps (v1)
    s = f"{wrap_zeroshot_psuedocode(sentence,example)}" # zero-shot + pseudocode
    # s = f"{wrap_one_shot_pseudocode(sentence,example)}" # one-shot pseudocode

    chat_completion = client.chat.completions.create(
        messages=[
 {
            'role': 'system',
            'content': f'''You are an intelligent programmer. You must complete the python function given to you by the user. And you must follow the format they present when giving your answer.
             You can only respond with actual code.'''
        },
            {
                "role": "user",
                "content": s, # 
            }
        ],
        model="gpt-3.5-turbo-0301",
    )
    label = chat_completion.choices[0].message.content
    if '<|start_of_completion|>' in label:
        label = label.split('<|start_of_completion|>')[1]
    return label




@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size, counter
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]

    # thoughts = load_json("./humaneval_actualpsuedocode_magicoder_reform_v1.json") # !NOTE for pseudocode examples
    thoughts = load_json ("./humaneval_steps_magicoder.json") # !NOTE for step by step examples
    example_thought = thoughts[counter]

    batch_completions = []
    for prompt in input_batch:
        ans = answer_query_gpt(prompt,example_thought)
        time.sleep(0.15)
        batch_completions.append(ans)

    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=api_key, 
        
    )

    parser = argparse.ArgumentParser(description="Eval model")
    parser.add_argument('--model_name', type=str, default='openai/gpt3.5-turbo')
    parser.add_argument('--length', type=int, default=100)
    args = parser.parse_args()
    print(args)

    num_samples_per_task = 5
    out_path = "results/" + args.model_name.split('/')[0] + "/humaneval_" + args.model_name.split('/')[1] + '_' + str(args.length) + ".jsonl"
    print ("Out path: ", out_path)
    os.makedirs("results/" + args.model_name.split('/')[0], exist_ok=True)

    run_eval(
        None,
        None,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        args.length,
        True,
    )




