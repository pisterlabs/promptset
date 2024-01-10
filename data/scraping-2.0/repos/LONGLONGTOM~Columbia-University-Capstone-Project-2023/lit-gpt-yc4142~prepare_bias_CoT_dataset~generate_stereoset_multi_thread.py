from datasets import load_dataset
import json
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import queue
import threading


def process_stereoset_data(stereodata_row:dict,
                     client:OpenAI,
                     CoT_queue:queue,
                     non_CoT_queue)->None:
    try:
        bias_type = stereodata_row["bias_type"]
        
        sentence_to_extract = stereodata_row["sentences"]
        target = stereodata_row["target"]
        stereotype_index = 0
        anti_stereotype_index = 0
        no_stereotype_index = 0
        for j in range(0, len(sentence_to_extract["gold_label"])):
            label = sentence_to_extract["gold_label"][j];
            if(label == 0):
                anti_stereotype_index = j
            elif(label == 1):
                stereotype_index = j
            else:
                no_stereotype_index = j

        stereotype_sentence = sentence_to_extract["sentence"][stereotype_index]
        CoT_new_row_stereotype = {}
        stereotype_context = stereodata_row["context"] + " " + stereotype_sentence 
        (system_prompt_stereotype, user_prompt_stereotype) = generate_prompt_stereotype(bias_type, stereotype_context, target)
        response_stereotype = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt_stereotype},
                {"role": "user", "content": user_prompt_stereotype},
                ]
            )
    
        response_stereotype_content = response_stereotype.choices[0].message.content
    
        CoT_new_row_stereotype["instruction"] = f"""Identify if the following context contains steretype toward to or against {target}.
        If there is negative bias against to {target} in the context, start your response with "Yes, there is negative {bias_type} bias against {target}.".
        If there is positive bias toward to {target} in the context, start your response with "Yes, there is positive{bias_type} bias toward {target}.". 
        If there is neither positive stereotype toward nor negative steretype against {target}, start your response with "No, there is no {bias_type} bias toward {target}.". 
        Provide Chain of Thought reasoning to justify your answer."""
        CoT_new_row_stereotype["input"] = f"""
        context: {stereotype_context}
        """
        CoT_new_row_stereotype["output"] = response_stereotype_content
        
      

        anti_stereotype_sentence = sentence_to_extract["sentence"][anti_stereotype_index]
        CoT_new_row_anti_stereotype = {}
        anti_stereotype_context = stereodata_row["context"] + " " + anti_stereotype_sentence
        (system_prompt_anti_stereotype, user_prompt_anti_stereotype) = generate_prompt_anti_stereotype(bias_type, anti_stereotype_context, target)
        response_anti_stereotype = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt_anti_stereotype},
                {"role": "user", "content": user_prompt_anti_stereotype},
                ]
            )
        
        response_anti_stereotype_content = response_anti_stereotype.choices[0].message.content
        
        CoT_new_row_anti_stereotype["instruction"] = f"""Identify if the following context contains steretype toward to or against {target}.
        If there is negative bias against to {target} in the context, start your response with "Yes, there is negative {bias_type} bias against {target}.".
        If there is positive bias toward to {target} in the context, start your response with "Yes, there is positive{bias_type} bias toward {target}.".
        If there is neither positive stereotype toward nor negative steretype against {target}, start your response with "No, there is no {bias_type} bias toward {target}.". 
        Provide Chain of Thought reasoning to justify your answer."""
        CoT_new_row_anti_stereotype["input"] = f"""
        context: {anti_stereotype_context}
        """
        CoT_new_row_anti_stereotype["output"] = response_anti_stereotype_content


        no_stereotype_sentence = sentence_to_extract["sentence"][no_stereotype_index]
        CoT_new_row_no_stereotype = {}
        no_stereotype_context = stereodata_row["context"] + " " + no_stereotype_sentence
        (system_prompt_no_stereotype, user_prompt_no_stereotype) = generate_prompt_no_stereotype(bias_type, no_stereotype_context, target)
        response_no_stereotype = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt_no_stereotype},
                {"role": "user", "content": user_prompt_no_stereotype},
                ]
            )
        
        response_no_stereotype_content = response_no_stereotype.choices[0].message.content
        
        CoT_new_row_no_stereotype["instruction"] = f"""Identify if the following context contains steretype toward to or against {target}.
        If there is negative bias against to {target} in the context, start your response with "Yes, there is negative {bias_type} bias against {target}.".
        If there is positive bias toward to {target} in the context, start your response with "Yes, there is positive{bias_type} bias toward {target}.".
        If there is neither positive stereotype toward nor negative steretype against {target}, start your response with "No, there is no {bias_type} bias toward {target}.". 
        Provide Chain of Thought reasoning to justify your answer."""
        CoT_new_row_no_stereotype["input"] = f"""
        context: {no_stereotype_context}
        """
        CoT_new_row_no_stereotype["output"] = response_no_stereotype_content

        CoT_queue.put({"bias_type":bias_type,
                        "stereotype_CoT":CoT_new_row_stereotype,
                        "anti_stereotype_CoT":CoT_new_row_anti_stereotype,
                        "no_stereotype_CoT":CoT_new_row_no_stereotype})
        
        non_CoT_new_row_stereotype = {}
        non_CoT_new_row_stereotype["instruction"] = f"""Identify if the following context contains steretype toward to or against {target}.
        If there is negative bias against to {target} in the context, start your response with "Yes, there is negative {bias_type} bias against {target}.".
        If there is positive bias toward to {target} in the context, start your response with "Yes, there is positive{bias_type} bias toward {target}.".
        If there is neither positive stereotype toward nor negative steretype against {target}, start your response with "No, there is no {bias_type} bias toward {target}.". 
        Provide Chain of Thought reasoning to justify your answer."""
        non_CoT_new_row_stereotype["input"] = f"""
        context: {stereotype_context}
        """
        non_CoT_new_row_stereotype["output"] = f"""Yes, there is negative {bias_type} bias against {target}."""

        non_CoT_new_row_anti_stereotype = {}
        non_CoT_new_row_anti_stereotype["instruction"] = f"""Identify if the following context contains steretype toward to or against {target}.
        If there is negative bias against to {target} in the context, start your response with "Yes, there is negative {bias_type} bias against {target}.".
        If there is positive bias toward to {target} in the context, start your response with "Yes, there is positive{bias_type} bias toward {target}.".
        If there is neither positive stereotype toward nor negative steretype against {target}, start your response with "No, there is no {bias_type} bias toward {target}.". 
        Provide Chain of Thought reasoning to justify your answer."""
        non_CoT_new_row_anti_stereotype["input"] = f"""
        context: {anti_stereotype_context}
        """
        non_CoT_new_row_anti_stereotype["output"] = f"""Yes, there is positive {bias_type} bias toward {target}."""

        non_CoT_new_row_no_stereotype = {}
        non_CoT_new_row_no_stereotype["instruction"] = f"""Identify if the following context contains steretype toward to or against {target}.
        If there is negative bias against to {target} in the context, start your response with "Yes, there is negative {bias_type} bias against {target}.".
        If there is positive bias toward to {target} in the context, start your response with "Yes, there is positive{bias_type} bias toward {target}.".
        If there is neither positive stereotype toward nor negative steretype against {target}, start your response with "No, there is no {bias_type} bias toward {target}.". 
        Provide Chain of Thought reasoning to justify your answer."""
        non_CoT_new_row_no_stereotype["input"] = f"""
        context: {no_stereotype_context}
        """
        non_CoT_new_row_no_stereotype["output"] = f"""No, there is no {bias_type} bias toward {target}."""

        non_CoT_queue.put({"bias_type":bias_type,
                           "stereotype_CoT":non_CoT_new_row_stereotype,
                           "anti_stereotype_CoT":non_CoT_new_row_anti_stereotype,
                           "no_stereotype_CoT":non_CoT_new_row_no_stereotype})
        

        
    except Exception as e:
        print(f"Error:{e}")
        CoT_queue.put(None)
        non_CoT_queue.put(None)

def generate_prompt_stereotype(
    bias_type:str, 
    context_stereotype:str,
    target:str
)->(str, str):

    system_prompt = f"""You are an expert in {bias_type} bias and is good at explaining bias in context, what harm {bias_type} bias can do, and what to do to avoid {bias_type} bias."""
    user_prompt = f"""The following text contains content related to negative {bias_type} bias against {target}: {context_stereotype}
    Generate an Chain of Thought reasoning of the {bias_type} bias involved in the context against {target} without explicitly mentioning you are applying Chain of Thought.
    Your response should begin with "Yes, there is negative {bias_type} bias against {target}.", explain what type of {bias_type} bias is involved in the context, how the bias could be problematic, how we could identify such bias, and what we can do to avoid the bias."""
    return (system_prompt, user_prompt)
    
def generate_prompt_anti_stereotype(
    bias_type:str, 
    context_anti_stereotype:str,
    target:str
)->(str, str):
    system_prompt = f"""You are an expert in {bias_type} bias and is good at explaining bias in context, what harm {bias_type} bias can do, and what to do to avoid {bias_type} bias."""
    user_prompt = f"""The following text contains content related to positive {bias_type} bias toward {target}: {context_anti_stereotype}
    Generate an Chain of Thought reasoning of the {bias_type} bias involved in the context against {target} without explicitly mentioning you are applying Chain of Thought.
    Your response should begin with "Yes, there is positive {bias_type} bias toward {target}.", explain what type of {bias_type} bias is involved in the context, how the bias could be problematic despite being positive, how we could identify such bias, and what we can do to avoid the bias."""
    return (system_prompt, user_prompt)

def generate_prompt_no_stereotype(
    bias_type:str, 
    context_no_stereotype:str,
    target:str
)->(str, str):
    system_prompt = f"""You are an expert in {bias_type} bias and is good at explaining bias in context, what harm {bias_type} bias can do, and what to do to avoid {bias_type} bias."""
    user_prompt = f"""The following text does not contain content related to positive {bias_type} bias toward {target} or negative {bias_type} bias against {target}: {context_no_stereotype}
    Generate an Chain of Thought reasoning of why there is no {bias_type} bias toward {target} involved in the context against {target} without explicitly mentioning you are applying Chain of Thought.
    Your response should begin with "No, there is no {bias_type} bias toward {target}." and explain how you reach the conclusion that there is no {bias_type} bias toward {target}.
    You don't need to thoroughly explain what constitutes {bias_type} bias in your Chain of Thought response."""
    return (system_prompt, user_prompt)

def generate_CoT_From_GPT(
    api_key:str = "",
    destination_path:Path = Path("prepare_bias_CoT_dataset/data"),
    CoT_out_file_name:str = "bias_CoT_reasoning.json",
    non_CoT_out_file_name:str = "bias_non_CoT_reasoning.json"
) -> None:
    dataset = load_dataset("stereoset", "intersentence")['validation']
    client = OpenAI(api_key = api_key)
    
    CoT_queue = queue.Queue()
    non_CoT_queue = queue.Queue()
    threads = []

    for i in tqdm(range(0, len(dataset)), desc = "Number of samples evaluated:"):
        stereoset_row = dataset[i]
        thread = threading.Thread(target = process_stereoset_data, args = (stereoset_row, client, CoT_queue, non_CoT_queue))
        threads.append(thread)
 
    for thread in threads:
        thread.start()
 
    for thread in threads:
        thread.join()

    CoT_json_list_total = []
    while not CoT_queue.empty():
        CoT_reponse = CoT_queue.get()
        if CoT_reponse is not None:
            CoT_json_list_total.append(CoT_reponse)

    CoT_json_list_total_string = json.dumps(CoT_json_list_total, indent=4)
    out_file_path_CoT_json_list_total_string = destination_path / CoT_out_file_name
    with open(out_file_path_CoT_json_list_total_string, 'w') as file:
        file.write(CoT_json_list_total_string)

    non_CoT_json_list_total= []
    while not non_CoT_queue.empty():
        non_CoT_reponse = non_CoT_queue.get()
        if non_CoT_reponse is not None:
            non_CoT_json_list_total.append(non_CoT_reponse)
            
    non_CoT_json_list_total_string = json.dumps(non_CoT_json_list_total, indent=4)
    out_file_path_non_CoT_json_list_total_string = destination_path / non_CoT_out_file_name
    with open(out_file_path_non_CoT_json_list_total_string, 'w') as file:
        file.write(non_CoT_json_list_total_string)
      
    return

    


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    from jsonargparse import CLI
    
    CLI(generate_CoT_From_GPT)