import openai
import tiktoken
import json
import re
import hashlib
from . import prompts

def classify_page(url, body):
    if 'wp-json' in url:
        return {"blogpost": "no"}
    print("classify_page called with url:", url)
    prompt = prompts.BLOGPOST_CLASSIFICATION_AND_EXTRACT_TITLE_DATE
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo") 

    # Limit the number of tokens for input to OpenAI
    num_tokens = len(enc.encode(body + prompt + url))
    if num_tokens > 1000:
        num_other_tokens = len(enc.encode(prompt + url))
        body = enc.decode(enc.encode(body)[:1000 - num_other_tokens])

    # full prompt for model
    full_prompt = f"{prompt} \n \n URL: {url} \n Body: {body}"

    # get the chatGPT response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt},
            ]
    )
    classification = response["choices"][0]["message"]["content"]
    num_input_tokens = response["usage"]["prompt_tokens"]
    num_output_tokens = response["usage"]["completion_tokens"]

    try:
        return json.loads(classification), num_input_tokens, num_output_tokens
    except:
        # sometimes, chatGPT adds extra unnecessary words around the JSON response. This should catch that.
        match = re.search(r'\{.*\}', classification)
        if match:
            return json.loads(match.group()), num_input_tokens, num_output_tokens
        else:
            raise ValueError(f"Unable to parse chatGPT response. The following: \n \n {classification} \n \n is not JSON")
    
def encode_url_for_rtdb(url):
    """
    Encode a url for use as a key in Firebase realtime database
    """
    return hashlib.sha256(url.encode()).hexdigest()


# def encode_url(url):
#     url = url.replace(":", '%AA')
#     url = url.replace(".", '%AB')
#     url = url.replace("$", '%AC')
#     url = url.replace("[", '%AD')
#     url = url.replace("]", '%AE')
#     url = url.replace("#", '%AF')
#     url = url.replace("/", '%AG')
#     return url

# def decode_url(url):
#     url = url.replace('%AG', "/")
#     url = url.replace('%AF', "#")
#     url = url.replace('%AE', "]")
#     url = url.replace('%AD', "[")
#     url = url.replace('%AC', "$")
#     url = url.replace('%AB', ".")
#     url = url.replace('%AA', ":")
#     return url
    




