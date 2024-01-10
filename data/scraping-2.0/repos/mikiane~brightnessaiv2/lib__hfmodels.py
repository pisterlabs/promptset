import requests
import openai
from huggingface_hub import InferenceClient



def query(payload, headers, api_url):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()



#### NE FONCTIONNE PAS/ A DEBUGUER ####
def stream_mistral(prompt, api_token="none", max_tokens=1024):
    client = openai.OpenAI(api_key=api_token)  # Create an OpenAI client with the API key

    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=[
            {'role': 'system', 'content': "Je suis un assistant"},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0,
        max_tokens=max_tokens,
        stream=True
    )

    # For each part of the response
    for chunk in response:
        # If the part contains a 'delta' and the 'delta' contains 'content'
        if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
            content = chunk['choices'][0]['delta']['content']  # Extract the content
            print(content)
            yield f"{content}"  # Yield the content as a string


#############################################################################################################################
##### NE FONCTIONNE PAS/ A DEBUGUER ####

def stream_hfllm(prompt, api_token, api_url, max_token, num_tokens=300):
    client = InferenceClient(api_url, token=api_token)
    for token in client.text_generation(prompt, max_new_tokens=max_token, stream=True):
        yield f"{token}"



#############################################################################################################################

