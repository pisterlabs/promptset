import os
import pickle
import openai
import numpy as np
from transformers import pipeline
import torch
from dotenv import load_dotenv
load_dotenv()


# TODO MODEL
# model="openai"
# model="local"
model="zephyr"

def openai_model():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = "https://api.openai.com/v1"

def local_model():
    openai.api_key = "empty"
    openai.api_base = "http://localhost:8000/v1"

local_model()
if model=="zephyr":
    pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16,
                    device_map="auto")

def gpt(history: list[dict]):
    l = [x['content'] for x in history]
    return '\n---\n'.join(l)
def wizard_coder(history: list[dict]):
    DEFAULT_SYSTEM_PROMPT = history[0]['content']+'\n\n'
    B_INST, E_INST = "### Instruction:\n", "\n\n### Response:\n"
    messages = history.copy()
    messages_list=[DEFAULT_SYSTEM_PROMPT]
    messages_list.extend([
        f"{B_INST}{(prompt['content']).strip()}{E_INST}{(answer['content']).strip()}\n\n"
        for prompt, answer in zip(messages[1::2], messages[2::2])
    ])
    messages_list.append(f"{B_INST}{(messages[-1]['content']).strip()}{E_INST}")
    return "".join(messages_list)

def zephyr(history: list[dict]):
    system_prompt = history[0]['content']
    user_prompt = history[1]['content']
    prompt_template = f'''<|system|>
{system_prompt}</s>
<|user|>
{user_prompt}</s>
<|assistant|>
    '''
    return prompt_template
def chat_completion(system_message, human_message):
    history = [{"role": "system", "content": system_message}, {"role": "user", "content": human_message}]


    if model=='openai':
        completion=openai.ChatCompletion.create(model='gpt-3.5-turbo',messages=history, temperature=0, max_tokens=500)
        answer = completion['choices'][0]['message']["content"]
    elif model=='local':
        prompt = wizard_coder(history)
        # print(type(prompt))
        completion=openai.Completion.create(model='gpt-3.5-turbo', prompt=prompt, temperature=0, max_tokens=500)
        answer = completion['choices'][0]['text']
    elif model=='zephyr':
        # prompt = zephyr(history)
        # print(prompt)
        # completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=history, temperature=0, max_tokens=500)
        # answer = completion['choices'][0]['message']["content"]
        prompt=pipe.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        outputs=pipe(prompt, max_new_tokens=500)
        answer=outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()

    return answer


# Directory containing the pickle files
pickle_directory = 'question_set'

output_file_path = 'output_zephyr_short.txt'  # File where you want to save the output

# Open the output file
# message="I am very new to Sawyer, is there any introductionary material?"
# message="How does the True and False boolean logic work?"
# message = "How do i get started with MoveIt?"
message = "How should I load motion planner on MoveIt?"
picklefile = "recursive_seperate_none_openai_embedding_2300.pkl"
path_to_pickle = os.path.join("/home/bot/localgpt/rag/pickle/", picklefile)
with open(path_to_pickle, 'rb') as f:
    data_loaded = pickle.load(f)
doc_list = data_loaded['doc_list']
embedding_list = data_loaded['embedding_list']
id_list = data_loaded['id_list']
history = [{"role": "user", "content": message}]
# OPENAI
openai_model()
query_embed = np.array(openai.Embedding.create(model="text-embedding-ada-002", input=gpt(history))['data'][0]['embedding'])
# model
local_model()
cosine_similarities = np.dot(embedding_list, query_embed)
indices = np.argsort(cosine_similarities)[::-1]
docs = doc_list[indices]
top_docs=docs[:3]
distances = cosine_similarities[:3]
top_id=id_list[:3]
print(distances)
for mod in ["zephyr", "openai"]:
    if mod=="zephyr":
        model="zephyr"
        local_model()
    elif mod=="openai":
        model="openai"
        openai_model()
    way="seperate"
    if way=="together":
        insert_document = ""
        system_message = ("Use the provided articles delimited by triple quotes to find an answer for the given question. If the answer cannot be found in the articles, answer the question without the articles")
        for docu, id in top_docs, top_id:
            insert_document += f"Articles: \"\"\"{id}\n{docu}\"\"\"\n"
        insert_document+= f"{message}\n"
        print(chat_completion(system_message,insert_document))
    if way=="seperate":
        insert_document = ""
        for docu, id in zip(top_docs, top_id):
            # system_message = ('Use the provided article delimited by triple quotes to answer questions. Your task is to judge if the article contain essential part to answer the provided question, answer with "False" or "True" at the end.')
            # system_message = ("Read the article provided within the triple quotes. Your task is to assess whether the article contains a necessary part of the information related to given question or instruction. After your evaluation, conclude your response with either 'True' or 'False' to indicate whether the article is sufficient for answering the question.")
            # system_message='Read the article provided within the triple quotes carefully. Your primary task is to evaluate whether the content of the article contain key information that is crucial for the response to the question or instruction. Answer with "True" or "False" at the end.'
            system_message = '"Review the article within the triple quotes. Determine if it is useful for answering the given instruction. Say "True!" if the article helps, or "False!" if it is not.'
            # system_message = 'First Response to the given question or instruction. Then, Determine if the Response and the article provided are related. Answer with "(True)" or "(False)" at the end.'
            docs=f"\"\"\"{id}\n{docu}\"\"\"\n"
            question = f"Instruction:{message}\n"
            human_message = (docs+question)
            # print(human_message)
            # print(human_message)
            response=chat_completion(system_message,human_message).replace("\n\n\n", '')
            print(response)
            print('--------------------------------------------------------------------------------')
            if "true!" in response.lower():
                print(True)
                insert_document += docs
            else:
                print(False)
        # insert_document+= f"{message}\n"
        print("inserted docs")
        print(insert_document)
        if not insert_document:
            insert_document=f'{message}'
            system_message="Answer the question."
            print(chat_completion(system_message,insert_document))
        else:
            insert_document+=f'Instruction: {message}'
            system_message="Answer the instruction with the given articles."
            print(chat_completion(system_message, insert_document))