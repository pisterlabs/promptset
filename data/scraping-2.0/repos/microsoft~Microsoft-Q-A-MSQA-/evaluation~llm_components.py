import openai
import os
import requests
from keybook import keys
from time import sleep

def few_shot_message_fusion(instruction, exampels):
    messages=[
        {"role":"system","content":instruction['instruction']},        
    ]

    for example in exampels:        
        messages.append({"role":"system name=example_user","content":f"Statement:{example['Statement']}\nEvidence:{example['Evidence']}"})
        messages.append({"role":"system name=example_assistant","content":f"Answer:{example['Answer']}"})
    return messages

def prompt_fusion(message_list):
    '''
    :param message_list: standard format for ChatCompletion API, e.g., [{"role": user, "content":xxxx}]
    :return: string format of messages for Completion API
    '''

    start_token = "<|im_start|>"
    end_token = "<|im_end|>"

    fuszed_prompt = ""

    for message in message_list:
        message_string = start_token + message["role"] + "\n" + message["content"] + "\n" + end_token + "\n"
        fuszed_prompt += message_string

    fuszed_prompt += "<|im_start|>assistant\n"
    fuszed_prompt=fuszed_prompt.strip()

    return fuszed_prompt



def get_oai_completion_gpt_unified(message_list, gpt_version, temperature=0, max_tokens=800, top_p=0):
    openai.api_type = "azure"
    openai.api_base = "https://cloudgpt.openai.azure.com/"
    # openai.api_version = "2022-12-01"
    openai.api_version = "2023-03-15-preview"
    # openai.api_key = os.getenv("OPENAI_KEY")  # get openai_key via env var
    openai.api_key=keys['oai_key']
    if gpt_version == 3.5:
        engine = "gpt-35-turbo-20220309"
    elif gpt_version == 4:
        engine = "gpt-4-20230321"
    else:
        assert False, "gpt_version should be 3.5 or 4"

    try: 
        response = openai.ChatCompletion.create(
            engine=engine,
            messages=message_list,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
            timeout=60
        )
        gpt_output = response['choices'][0]['message']['content']
        return gpt_output
    except requests.exceptions.Timeout:
        # Handle the timeout error here
        print("The OpenAI API request timed out. Please try again later.")
        return None
    except openai.error.APIConnectionError as e:
        print(f"The OpenAI API connection failed: {e}")
        sleep(3)
        return get_oai_completion_gpt_unified(message_list, gpt_version)
    except openai.error.InvalidRequestError as e:
        # Handle the invalid request error here
        print(f"The OpenAI API request was invalid: {e}")
        return None
    except openai.error.Timeout as e:
        print(f"The OpenAI API read timed out: {e}")
        sleep(3)
        return get_oai_completion_gpt_unified(message_list, gpt_version)  
    except openai.error.RateLimitError as e:
        print("Token rate limit exceeded. Retrying after 3 second...")
        sleep(3)
        return get_oai_completion_gpt_unified(message_list, gpt_version)  
    except openai.error.APIError as e:
        if "The operation was timeout" in str(e):
            # Handle the timeout error here
            print("The OpenAI API request timed out. Please try again later.")
            sleep(3)
            return get_oai_completion_gpt_unified(message_list, gpt_version)           
        # elif "Requests to the Creates a completion for the chat message Operation under Azure OpenAI API version 2023-03-15-preview have exceeded token rate limit" in str(e):
        #     # Handle the rate limit error here
        #     print("Token rate limit exceeded. Retrying after 3 second...")
        #     sleep(3)
        #     return get_oai_completion_gpt4(message_list) 
        else:
            # Handle other API errors here
            print(f"The OpenAI API returned an error: {e}")
            return None




