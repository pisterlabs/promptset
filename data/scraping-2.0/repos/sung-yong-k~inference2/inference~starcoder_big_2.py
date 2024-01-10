import os
import openai
import shutil
from cppcheck_test import*
import pandas as pd
import json
import time
import re
from infer_test import*
from transformers import AutoTokenizer, pipeline, logging, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
import torch
from hf_hub_ctranslate2 import GeneratorCT2fromHfHub


print("install openai and cppcheck and create production before before")
#key = input("Enter a key: ")
#openai.api_key = key


def balance_c_code(input_code):
    opening_braces = 0
    closing_braces = 0
    balanced_code = ""

    for char in input_code:
        if char == '{':
            opening_braces += 1
        elif char == '}':
            closing_braces += 1

        balanced_code += char

    if opening_braces > closing_braces:
        # Add the necessary closing braces
        balanced_code += '}' * (opening_braces - closing_braces)

    return balanced_code




def generation(prompt, llm_code, vuln_list):
    prompt_2 = "Complete this code \n"+prompt
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    temperature= 0,
    messages=[
    {"role": "system", "content": "You are a coding assistant and help the user."},
    {"role": "user", "content": "When I say complete a code, I want you to complete it and give back the function from the beginning."},
    {"role": "assistant", "content": "Of course, I'd be happy to help you complete a code snippet. Please provide the initial function or code snippet, and let me know what specific task or functionality you'd like me to assist you with."},
    {"role": "user", "content": "Also, before giving the prompt, I will tell you what kind of vulnerabilities this prompt can produce."},
    {"role": "assistant", "content": "Certainly, please go ahead and describe the kind of vulnerabilities or issues you're concerned about with the code prompt you have in mind. Providing this information will help me better understand your requirements and ensure that the generated code is secure and free from common vulnerabilities."},
    {"role": "user", "content": "Here is the list ordered by their severity. "+vuln_list+"\n Complete the following code without generating the vulnerabilities: \n"+prompt}
    ]
     )

    #print(response)
    if 'choices' in response:
        x = response['choices']
        #print( x[0]['message']['content'])
        answer = x[0]['message']['content']
        code= extract_substring(answer,"#include","}")
        return code

def generationbis(prompt, llm_code, vuln_list):
    prompt_2 = "Complete this code \n"+prompt
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    temperature= 0,
    messages=[
    {"role": "system", "content": "You are a coding assistant and help the user."},
    {"role": "user", "content": "When I say complete a code, I want you to complete it and give back the function from the beginning."},
    {"role": "assistant", "content": "Sure! Please provide the code snippet that you want me to complete."},
    {"role": "user", "content": prompt_2},
    {"role": "assistant", "content": llm_code},
    {"role": "user", "content": "The code has different vulnerabilities. Here is the list ordered by their severity. "+vuln_list+"\n Complete the following code from scratch without generating the same vulnerabilities: \n"+prompt}
    ]
     )

    #print(response)
    if 'choices' in response:
        x = response['choices']
        #print( x[0]['message']['content'])
        answer = x[0]['message']['content']
        code= extract_substring(answer,"#include","}")
        return code



def generation2(prompt, gpt_msg, user_prompt):
    prompt2 = "Complete this code \n"+prompt
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    temperature= 0,
    messages=[
    {"role": "system", "content": "You are a coding assistant and help the user."},
    {"role": "user", "content": "When I say complete a code, I want you to complete it and give back the function from the beginning."},
    {"role": "assistant", "content": "Sure! Please provide the code snippet that you want me to complete."},
    {"role": "user", "content": prompt2},
    {"role": "assistant", "content": gpt_msg},
    {"role": "user", "content": "The code does not satsify my expectation, please recomplete the code:\n"+ prompt}
    ]
     )

    #print(response)
    if 'choices' in response:
        x = response['choices']
        #print( x[0]['message']['content'])
        answer = x[0]['message']['content']
        code= extract_substring(answer,"#include","}")
        return code

def extract_substring(s, start_str, end_str):
    """
    Extract a substring from s that starts with start_str and ends with end_str.
    """
    start_index = s.find(start_str) + len(start_str)
    end_index = s.rfind(end_str, start_index)
    if start_index < 0 or end_index < 0:
        return ""
    return s[start_index-8:end_index+1]

def extract_response(text):
    keyword = "### Response:"
    
    # Check if the keyword is in the text
    if keyword in text:
        # Find the index of the keyword and extract everything after it
        return text[text.index(keyword) + len(keyword):].strip()
    else:
        return None

model_name = "michaelfeil/ct2fast-starcoder"
model = GeneratorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name,
        device="cuda",
        compute_type="int8_float16",
        # tokenizer=AutoTokenizer.from_pretrained("{ORG}/{NAME}")
)



command_list=["starcoder"]
v = ["V1"]

for root, dirs, files in os.walk("prompt_final"):
    for file_name in files:
        print("file name is: ")
        print(file_name)
        
        if file_name != ".DS_Store"  :
            generation_type=0
            start = time.time()
            file_path = os.path.join(root, file_name)
            input = open("prompt_final/"+file_name, "r",encoding='utf-8')
            line = input.readlines()
            input.close()
            total_lines = len(line)
            copy_lines = int(total_lines * 1)
            text2 = "".join(line[:copy_lines])
            text = "Complete the code:\n"+text2     

            outputs = model.generate(
                text=[text],
                max_length=300,
                include_prompt_in_result=False     
                ,temperature=0.1
            )

            print(outputs)
            # inputs = tokenizer.encode(text, return_tensors="pt").to(device)
            # outputs = model.generate(inputs=inputs,do_sample=True,temperature=0.1,max_new_tokens=500,top_k= 50)
            # code = extract_substring(tokenizer.decode(outputs[0]),"#include","}")
            code = outputs[0]
            print(code)
            end_time=time.time()-start
            temps = str(end_time)   

            input = open("Result/starcoder_time_" +temps+"_"+ file_name, "w",encoding='utf-8')  
 
            input.write(code)
            input.close()
           
