import os
import requests
import openai

MAX_CHAT_SIZE = 4000

openai_api_key = os.getenv('OPENAI_API_KEY')

# for bulk openai message, no stream
def chat_openai(question="Tell me to ask you a prompt", website="", chat_history=[]):
    # define prompt
    prompt = question
    if website:
        prompt = "Given this information: " + website[:MAX_CHAT_SIZE - len(question)] + ", respond conversationally to this prompt: " + question

    # define message conversation for model
    if chat_history:
        messages = chat_history
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant, a large language model trained by OpenAI. Answer as concisely as possible. If you're unsure of the answer, say 'Sorry, I don't know'"},
        ]
    messages.append({"role": "user", "content": prompt})

    # create the chat completion
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    text_answer = completion["choices"][0]["message"]["content"]

    # updated conversation history
    messages.append({"role": "assistant", "content": text_answer})

    return text_answer, messages

def chat_openai_stream(question="Tell me to ask you a prompt", website="", chat_history=[]):
    print("chatting openai stream in routes!")

    if website:
        # prompt = "Given this information: " + website[:MAX_CHAT_SIZE - len(question)] + ", respond conversationally to this prompt: " + question
        prompt = "Given the following information, answer the following question regardless of whether it's related to this text or not: " +  website[:MAX_CHAT_SIZE - len(question)]
    # define message conversation for model
    
    
    if chat_history:
        messages = chat_history
    else:
        messages = [
            # {"role": "system", "content": "You are a helpful assistant, a large language model trained by OpenAI. Answer as concisely as possible. If you're unsure of the answer, say 'Sorry, I don't know'"},
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "user", "content": question})

    
    # create the chat completion
    reqURL = 'https://api.openai.com/v1/chat/completions'
    reqHeaders = {
        'Accept': 'text/event-stream',
        'Authorization': 'Bearer ' + openai_api_key
    }
    reqBody = {
        'model': 'gpt-3.5-turbo',
        'messages': messages,
        'stream': True,
        'temperature': 0.8,
        'max_tokens' : 2048,
    }

    # updated conversation history
    response = requests.post(reqURL, stream=True, headers=reqHeaders, json=reqBody)

    return response, messages

# FOR TESTING
# if __name__ == "main":
    # client, messages = chat_openai_stream(question = "Are you still here?", website="The most famous Egyptian pyramids are those found at Giza, on the outskirts of Cairo. Several of the Giza pyramids are counted among the largest structures ever built.[9] The Pyramid of Khufu is the largest Egyptian pyramid. It is the only one of the Seven Wonders of the Ancient World still in existence, despite its being the oldest wonder by about 2,000 years.[10]")

    # for event in client.events():
    #     if event.data != '[DONE]' and 'content' in json.loads(event.data)['choices'][0]["delta"]:
    #         # print(event.data)
    #         print(json.loads(event.data)['choices'][0]["delta"]["content"], end="", flush=True)
