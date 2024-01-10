import pandas as pd
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, VLLM
from langchain.chains import LLMChain
import tiktoken
from langchain.memory import ConversationBufferMemory
import pickle
import os
import torch
from sklearn.utils import shuffle

os.environ["OPENAI_API_KEY"] = ""
os.environ["TIKTOKEN_CACHE_DIR"] = "/home/zw3/Assignment-3-ANLP/tmp"

torch.cuda.empty_cache()
model_id = "/data/datasets/models/huggingface/meta-llama/Llama-2-70b-chat-hf"
llm = VLLM(model=model_id, tensor_parallel_size=4, gpu_memory_utilization=0.95, top_k=1,
           stop=['\n', '.', '<\s>', 'nHuman', 'Human'], max_new_tokens=7)

tasks = ['0','1','2','3']

for task in tasks:
    rev_label = False
    shuffle = False
    file_name = "mbti_test_zero_standard_{}.csv".format(task)

    with open('data/Kaggle/test.pkl', 'rb') as f:
        data = pickle.load(f)
    def process_data(data):
        poster = data['posts_text']
        label = data['annotations']
        label_lookup = {'E': 1, 'I': 0, 'S': 1, 'N':0, 'T': 1, 'F': 0, 'J': 1, 'P':0}
        persona_lookup = {}
        poster_data = [{'posts': t, 'label0': label_lookup[list(label[i])[0]],
                'label1': label_lookup[list(label[i])[1]],'label2': label_lookup[list(label[i])[2]],
                'label3': label_lookup[list(label[i])[3]]} for i,t in enumerate(poster)]
        return poster_data
    poster_data = process_data(data)
    texts = [item['posts'] for item in poster_data]
    labels = [item['label{}'.format(task)] for item in poster_data]
    
    
    if task == '0':
        if not rev_label:
            trait_choice = 'A: "Extraversion" or B: "Introversion"'
        else:
            trait_choice = 'A: "Introversion" or B: "Extraversion"'
    elif task == '1':
        if not rev_label:
            trait_choice = 'A: "Sensing" or B: "Intuition"'
        else:
            trait_choice = 'A: "Intuition" or B: "Sensing"'
    elif task == '2':
        if not rev_label:
            trait_choice = 'A: "Thinking" or B: "Feeling"'
        else:
            trait_choice = 'A: "Feeling" or B: "Thinking"'
    elif task == '3':
        if not rev_label:
            trait_choice = 'A: "Judging" or B: "Perceiving"'
        else:
            trait_choice = 'A: "Perceiving" or B: "Judging"'

    def cteat_agent(trait_choice):
        perfix_temp = '<s>[INST] <<SYS>>\nYou are an AI assistant who specializes in text analysis. You will complete a text analysis task. '\
                'The task is as follows: according to a set of posts written by an author, '\
                'predicting whether the author is {}.'.format(trait_choice)
        Prompt_temp = perfix_temp + \
                '\n<</SYS>>\n\nAUTHOR\'S POSTS: {context}\n' + \
                'The author is {}. Provide a choice in the format: "CHOICE: <A/B>" and do not give the explanation. [/INST]'.format(trait_choice)
        prompt = PromptTemplate(
            input_variables=["context"],
            template=Prompt_temp)
        agent = LLMChain(llm=llm, prompt=prompt, verbose=False)
        return agent

    

    def shuffle_post(posts_list):
        shuffle_index = shuffle([i for i in range(len(posts_list))], random_state=0)
        shuffle_post_list = [posts_list[j] for j in shuffle_index]
        return shuffle_post_list

    if not os.path.exists(file_name):
        rest_sample = 0
        head = {"text": [], 'gold': [], 'answers': []}
        data = pd.DataFrame(head)
        data.to_csv(file_name, mode='w', index=False, header=True)
    else:
        data = pd.read_csv(file_name)
        rest_sample = len(list(data['gold']))
    print('Done: {}/{}'.format(rest_sample, len(texts)))
    tokenizer = tiktoken.get_encoding("cl100k_base")
    agent = cteat_agent(trait_choice)
    for i in range(rest_sample, len(texts)):
        print('Done: {}/{}'.format(i, len(texts)))
        posts = ''
        count = 1
        if shuffle:
            posts_list = shuffle_post(texts[i])
        else:
            posts_list = texts[i]
        for j in range(len(posts_list)):
            if len(posts_list[j])>10:
                post = tokenizer.decode(tokenizer.encode(posts_list[j].replace('{', '').replace('}', ''))[:80])
                posts += 'Post{}: {}; '.format(count,post)
                count += 1
        print(posts)
        answer = agent.predict(context=posts)
        print(answer)
        api_data = pd.DataFrame([[posts, labels[i], answer]])
        api_data.to_csv(file_name, mode='a', index=False, header=False)
