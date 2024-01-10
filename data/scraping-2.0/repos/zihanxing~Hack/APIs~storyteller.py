import openai
import os
import json
import random
import re
import requests
import io
from PIL import Image

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.getenv('OPENAI_API_KEY')



def get_completion(prompt, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=.5, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_imagePrompt(text):
    prompt = f"""
    Based on the following sentences, provide me a farly short prompt for image generation. It is generated for kids. \
    example return: pixel art, a cute corgi, simple, flat colors \
    ```{text}```
    """
    response = get_completion(prompt)
    # print()
    # print("Image Prompt:", response)
    # print()
    return response



def ask_openai(messages, temperature=0.5):
    try:
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            messages=messages,
            temperature=temperature,
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        return "I'm sorry, but I can't answer that question."

def ask_question():
    json_file = open("./static/storyQ.json", "r")
    pre_questions = json.load(json_file)
    q_list = pre_questions['questions']
    json_file.close()

    print("Hello! Welcome to GabbyGarden! I am Gab. To listen to a story, please answer me the question.")

    conversation_log = [] # to maintain the conversation context

    context = "You are a chatbot telling stories to young kids.\n\
        I will provide you a question and answer from the kid. \
        Based on the answer, you will tell a funny story to the kid. \n\
        As I am going to split the story into 3 parts, please provide a very short story with 3 short paragraphs respectively."

    conversation_log.append({"role": "system", "content": context})

    q = random.choice(q_list)

    return q, conversation_log

def story_trunks(q, answer, conversation_log):
    answer = answer.replace("\n", " ")
    
    conversation_log.append({"role": "assistant", "content": q})
    conversation_log.append({"role": "user", "content": answer})

    bot_response = ask_openai(conversation_log)

    conversation_log.append({"role": "assistant", "content": bot_response})

    # Deal with the return story
    sentences = re.split(r'(?<=[.!?]) +', bot_response)

    # Group sentences by threes and store them in a list
    grouped_sentences = [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]

    prompts = []
    for i in range(len(grouped_sentences)):
        prompt = get_imagePrompt(grouped_sentences[i])
        prompts.append(prompt)
    
    # merge each pair[grouped_sentences,prompts] into a list and return
    res=[]
    for g,p in zip(grouped_sentences,prompts):
        temp=[]
        temp.append(g)
        temp.append(p)
        res.append(temp)
        
    return res


def main():
    question, log = ask_question()
    input_answer = input(question)

    result = story_trunks(question, input_answer, log)
    # print(result)
    # print(result[0][0])


if __name__ == '__main__':
    main()




