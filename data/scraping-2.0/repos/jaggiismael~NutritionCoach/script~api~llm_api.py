import openai

class LlmApi:
    def __init__(self, key):
        
        openai.api_base = "https://api.deepinfra.com/v1/openai"
        openai.api_key = key
        MODEL_DI = "meta-llama/Llama-2-70b-chat-hf"

    #Function to send requests to the DeepInfra API. Return value: Generated response from the LLM in text form
    def send_prompt(self, messages, temperature): 
        chat_completion = openai.ChatCompletion.create(
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=messages,
        stream=True,
        temperature=temperature,
        max_token=100
        )

        answer = ""
        for event in chat_completion:
            if 'content' in event.choices[0]['delta']:
                answer += event.choices[0]["delta"]["content"]

        return answer
    

