from secrets_openai import SECRET_KEY_OPENAI
import pandas as pd
import argparse
from tqdm import tqdm
import requests
import openai
import os
import json
import time
import icecream as ic
openai_api_key = SECRET_KEY_OPENAI
openai.api_key = openai_api_key
# Reference for library error type and method to handle
# https://platform.openai.com/docs/guides/error-codes/python-library-error-types

def create_chat_completion(model, system_message, user_prompt, val_file, temperature, max_tokens):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
        {"role": "system", "content": system_message},
        {"role": "system", "name":"example_user", "content": str(val_file['sent'][0])},
        {"role": "system", "name": "example_assistant", "content": str(val_file['facts'][0])},
        {"role": "system", "name":"example_user", "content": str(val_file['sent'][1])},
        {"role": "system", "name": "example_assistant", "content": str(val_file['facts'][1])},
        {"role": "system", "name":"example_user", "content": str(val_file['sent'][2])},
        {"role": "system", "name": "example_assistant", "content": str(val_file['facts'][2])},
        {"role": "system", "name":"example_user", "content": str(val_file['sent'][3])},
        {"role": "system", "name": "example_assistant", "content": str(val_file['facts'][3])},
        {"role": "system", "name":"example_user", "content": str(val_file['sent'][4])},
        {"role": "system", "name": "example_assistant", "content": str(val_file['facts'][4])},
        {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        ) 
        return response
    except openai.error.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        time.sleep(10)
        return create_chat_completion(model, system_message, user_prompt, temperature, max_tokens)
    except openai.error.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")
        time.sleep(10)
        return create_chat_completion(model, system_message, user_prompt, temperature, max_tokens)
    except openai.error.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
        # Exponential backoff logic can be added here.
        # But for now, we will just wait for 60 seconds before retrying as is based on TPM.
        # https://platform.openai.com/docs/guides/rate-limits/what-are-the-rate-limits-for-our-api
        time.sleep(60)
        return create_chat_completion(model, system_message, user_prompt, temperature, max_tokens)
    except openai.error.AuthenticationError as e:
        print(f"Authentication error with OpenAI API: {e}")
        # Handle authentication error here. You may need to regenerate your API key.
        return None
    except openai.error.InvalidRequestError as e:
        print(f"Invalid request error with OpenAI API: {e}")
        # Handle invalid request error here. You may need to check your request parameters.
        return None
    except openai.error.ServiceUnavailableError as e:
        print(f"Service unavailable error with OpenAI API: {e}")
        time.sleep(30)
        return create_chat_completion(model, system_message, user_prompt, temperature, max_tokens)
    except requests.exceptions.ReadTimeout as e:
        print(f"Network request took too long: {e}")
        time.sleep(60)  # Wait for 60 seconds before retrying
        return create_chat_completion(model, system_message, user_prompt, temperature, max_tokens)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        time.sleep(600)
        return create_chat_completion(model, system_message, user_prompt, temperature, max_tokens)
    
def chat_api(prompt_context, LANG, val_file, system_message="", temperature=0):
    system_message = f"You must extract all facts in English from the following {LANG} sentence. A fact consists of a relation and tail entity present in the sentence. Return the extracted facts in the form of a list of lists."
    # Read from prompt-esconv-strategy.txt as text
    prompt = prompt_context
    # Ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    # Should work with "gpt-4" if you have access to it.
    ANNOTATING_MODEL = "gpt-4"
    # response = openai.ChatCompletion.create(
    #     model=ANNOTATING_MODEL,
    #     messages=[
    #         {"role": "system", "content": system_message},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=temperature,
    #     max_tokens=900,
    # )
    # print(prompt)
    response = create_chat_completion(ANNOTATING_MODEL, system_message, prompt, val_file, temperature, 1000)
    response_content = response['choices'][0]['message']['content']
    return response_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input parameters for gpt-4')
    parser.add_argument('--language', help='language of the sentences')
    parser.add_argument('--val_csv', help='path to the validation csv files')
    parser.add_argument('--test_csv', help='path to the test csv files')
    parser.add_argument('--output_csv', help='path to store the output csv containing the OpenAI results')
    args = parser.parse_args()
    languages_map = {
    'bn': {"label": "Bengali"},
    'en': {"label": "English"},
    'hi': {"label": "Hindi"},
    'or': {"label": "Odia"},
    'pa': {"label": "Punjabi"},
    'ta': {"label": "Tamil"},    
}
    lang_code = args.language
    LANG = languages_map[lang_code]['label']
    val_csv_path = args.val_csv
    test_csv_path = args.test_csv
    val_file = pd.read_csv(val_csv_path)
    test_file = pd.read_csv(test_csv_path)
    output_csv_path = args.output_csv
    val_file = val_file[0:5]
    #change
    #test_file = test_file[0:5]
    output = []
    for i in tqdm(range(len(test_file))):
        response_content = chat_api(str(test_file['sent'][i]),LANG, val_file)
        output.append({
                'sent': str(test_file['sent'][i]),
                'facts': str(test_file['facts'][i]),
                'pred_facts': response_content,
            })
        current_df = pd.DataFrame([output[-1]])
        current_df.to_csv(output_csv_path, mode='a', header=not i, index=False)
    print("Done!")
