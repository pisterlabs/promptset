import os
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
from core import filter_code, run_eval_mbpp, fix_indents
import os, json 
import torch
import time
import openai
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

openai.key = 'SET OPENAI KEY'
# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = "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"

#  ZERO SHOT PSEUDOCODE
def zero_shot_pseudocode(instruction,example):
    PROMPT = f'''
You must complete the python function I give you. When doing so, you must write the completion in the following form:
${{ORIG_FUNCTION}}
    ${{INSERT_COMPLETION}}


ORIG_FUNCTION=
{instruction}

You are given the following psuedocode to help you formulate the python function definition that should come after. 
PSEUDOCODE=
{example}

Once more, please follow the template by responding with the python definition then writing the completion for it. Follow python rules.
RESPONSE=
'''
    return PROMPT

# ONE SHOT + STEPS PROMPT
def one_shot_steps(instruction,example):
    PROMPT = f'''
You must complete the python function I give you. When doing so, you must write the completion in the following form:
${{ORIG_FUNCTION}}
    ${{INSERT_COMPLETION}}

ORIG_FUNCTION=
You are an expert Python programmer, and you need to respond with code to complete is your task: Given two binary strings a and b, return their sum as a binary string.
Your code should pass these tests:
assert binary_sum("11", "1") == "100"

    
SOLVING PROCESS=
1. Reverse the input binary strings 'a' and 'b' to prepare for right-to-left addition.
2. Iterate through both strings simultaneously, calculating the sum of current bits along with the carry.
3. Store the resulting sum bit in 'result' and update the carry.
4. fter the loop, check if there's any remaining carry and append it to the 'result'. Finally, return the 'result' as the binary sum.

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

ORIG_FUNCTION=
{instruction}

SOLVING PROCESS=
{example}

Once more, please follow the template by repeating the original function then writing the completion.
RESPONSE=
''' # v1 --this is the current version--



    PROMPT_v0 = f'''
You must complete the python function I give you. When doing so, you must write the completion in the following form:
${{ORIG_FUNCTION}}
    <|start_of_completion|>
    ${{INSERT_COMPLETION}}

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

ORIG_FUNCTION=
{instruction}

SOLVING PROCESS=
{example}

Once more, please follow the template by repeating the original function, including the <|start_of_completion|> token, then writing the completion.
RESPONSE=
''' # v0


    return PROMPT

# ONE SHOT + PSUEDOCODE PROMPT
def one_shot_pseudocode(instruction,example):
    PROMPT = f'''
You must complete the python function I give you. When doing so, you must write the completion in the following form:
${{ORIG_FUNCTION}}
    ${{INSERT_COMPLETION}}

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

ORIG_FUNCTION=
{instruction}

SOLVING PROCESS=
{example}

Once more, please follow the template by repeating the original function definition then writing the completion.
RESPONSE=
''' # v1 --this is the current version--
    return PROMPT

#ZERO SHOT WITH STEPS/CODE (modify accordingly with the thought)
def wrap_with_steps(instruction,example):

    P_for_steps = f'''
    You must complete the python function I give you. When doing so, you must write the completion in the following form:
    ${{ORIGINAL_FUNCTION}}
    <|start_of_completion|>
    ${{INSERT_COMPLETION}}

    Be sure to use the same indentation I specified. Furthermore, you may only write your response in code.
    You have been also provided a solving process to make your solutions more accurate.

    ORIGINAL_FUNCTION=
    {instruction}

    SOLVING PROCESS=
    {example}

    Once more, please follow the template by repeating the original function, including the <|start_of_completion|> token, then writing the completion.
    '''

    P_for_code = f'''
    You must complete the python function I give you. When doing so, you must write the completion in the following form:
    ${{ORIGINAL_FUNCTION}}
    <|start_of_completion|>
    ${{INSERT_COMPLETION}}

    Be sure to use the same indentation I specified. Furthermore, you may only write your response in code.
    You have been also provided a solving process to make your solutions more accurate.

    ORIGINAL_FUNCTION=
    {instruction}

    SOLVING PROCESS=
    {example}

    Once more, please follow the template by repeating the original function, including the <|start_of_completion|> token, then writing the completion.
    '''

    return P


# ZERO SHOT BASELINE
def wrap_code_template_baseline(instruction):
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


# ZERO BASELINE PROMPT VARIANT  
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

# BASELINE PROMPT VARIANT ( NO STEPS OR PSEUDOCODE)
def wrap_base_template(instruction):
    P = f'''
{instruction}
'''
    return P


def answer_query_gpt(sentence,example,counter=0):
    """Answers a query based on the prompt
    Args:
        sentence (str) : Prompt to answer
    Returns:
        label (str) : Answer to the query
    """
    # s = f"{wrap_code_template(sentence)}" # zero shot  variant
    # s = f"{wrap_code_template_baseline(sentence)}" # zero shot baseline variant

    # s = f"{wrap_with_steps(sentence,example)}" # zero shot example + step by step/ pseudocode

    # s = f"{one_shot_steps(sentence,example)}" # one shot example + step by step 

    # s = f"{zero_shot_pseudocode(sentence,example)}" # zero shot pseudocode

    s = f"{one_shot_pseudocode(sentence,example)}" # one shot pseudocode

    

    #if none is selected above, we are using the base template, without any steps or pseudocode
    # s = f"{wrap_base_template(sentence)}" # base template

    if counter == 11:
        print("The prompt looks like: \n")
        print(s)
        print(sentence)
    # s = f"{wrap_code_advanced(sentence,example)}"
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
    # if '<|start_of_completion|>' in label:
    #     label = label.split('<|start_of_completion|>')[1]
    return label

def load_json(file_path = "./mbpp_examples_magicoder_reform_v1.json"):
    with open(file_path, "r") as json_file:
        data_dict = json.load(json_file)
    
    list_prompts = []
    for k,v in data_dict.items():
        list_prompts.append(v)
    return list_prompts


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size, counter
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    # print ("Input: ", input_batch)
    counter = counter + 10
    # thoughts = load_json("./mbpp_examples_magicoder_reform_v1.json") # for loading steps
    thoughts = load_json("./mbpp_actualpsuedocode_magicoder_reform_v1.json") # for loading pseudocode
    example_thought = thoughts[counter]


    batch_completions = []
    for prompt in input_batch:
        ans = answer_query_gpt(prompt,example_thought,counter)
        time.sleep(0.13)
        batch_completions.append(ans)

    return [completion for completion in batch_completions]
    # return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    parser = argparse.ArgumentParser(description="Eval model")
    parser.add_argument('--model_name', type=str, default='openai/gpt3.5-turbo')
    parser.add_argument('--length', type=int, default=100)
    args = parser.parse_args()
    print(args)

    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=api_key, 
        
    )

    num_samples_per_task = 5
    out_path = "results/" + args.model_name.split('/')[0] + "/mbpp_" + args.model_name.split('/')[1] + '_' + str(args.length) + ".jsonl"
    os.makedirs("results/" + args.model_name.split('/')[0], exist_ok=True)

    run_eval_mbpp(
        None,
        None,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        args.length,
        True,
    )

    print ("Out path: ", out_path)
