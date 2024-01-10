import replicate
import requests
def llama(prompt):
    output = replicate.run(
    "meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0",
    input={"prompt": prompt,"max_new_tokens":1000}
)
# The meta/llama-2-7b-chat model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
    o = ""
    for item in output:
    ## https://replicate.com/meta/llama-2-7b-chat/versions/13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0/api#output-schema
       
        o+=item
    return o
def mistral(prompt):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
	
    output = query({
    "inputs": prompt,"parameters" : {"max_new_tokens": 256}
    })
    
    return output[0]['generated_text']
def mistralV2(prompt):
    try:
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
        headers = {"Authorization": "Bearer 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"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
        
        output = query({
        "inputs": prompt,"parameters" : {"max_new_tokens": 256}
        })
        
        return output[0]['generated_text']
    except:
        return output
    
import openai

def chatgpt(prompt, max_tokens=1000, temperature=0.7):
    api_key = "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"  # Replace with your OpenAI API key
    openai.api_key = api_key

    response = openai.Completion.create(
        engine="text-davinci-002",  # Use the appropriate engine for ChatGPT
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response.choices[0].text

def obs(prompt):
    

    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
        
    output = query({
        "inputs": prompt,"parameters" : {"max_new_tokens": 256}
    })
    return output[0]['generated_text']

# print(mistralV2("What is reinforcement Learning?"))
# def openAi(prompt):







#print(llama("Explain the working of photosynthesis briefly"))
