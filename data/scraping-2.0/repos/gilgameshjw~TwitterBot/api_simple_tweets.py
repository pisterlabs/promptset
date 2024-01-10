
import openai
import time
import json
import yaml
import os
import tqdm 
import random

from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings 
from flask import Flask, render_template, request
import numpy as np
import pickle as pkl


app = Flask(__name__)

# read config.yaml file
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

api_key = config["openai"]["openai_key"]
openai.api_key = api_key
openai_engine = config["openai"]["openai_engine"]
model_name = config["openai"]["model_name"]
twitter_handle = config["twitter"]["twitter_handle"]

# chatbot parameters
search_mode = config["chatbot"]["search_mode"]
run_mode = config["chatbot"]["run_mode"]
similarity_threshold = config["chatbot"]["similarity_threshold"]

# run_mode = "light"
if run_mode == "light":
    openai_engine = openai_engine
    print("--log:: run_mode is light, using openai_engine: ", openai_engine)
    similarity_threshold = 0.0

else:
    openai_engine = model_name
    print("--log:: run_mode is full, using model_name: ", model_name)


# check if numpy vector file exists
# if not, create it
vector_file = "data/vectorised_prompts_"+twitter_handle #+".npy"
data_file = "data/parsed_twitter_data_"+twitter_handle+"_prepared.jsonl"

# load data
with open(data_file) as f:
    data = f.readlines()
data = [json.loads(line[:-1]) for line in data]

# massage historical data for retrieval
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# build model and embedding once if file not present
if not os.path.exists(vector_file+".pkl"):

    # massage historical data for retrieval
    dict_data = dict([(d["prompt"], d["completion"]) for d in data])
    prompts = list(set([d["prompt"] for d in data]))
    v_hist_data = [dict_data[p] for p in prompts]
    # embed prompts
    m_prompts = \
        np.array([embeddings.embed_query(p) for p in tqdm.tqdm(prompts)])
    # write m_prompts into pickle file
    data_embeddings = {"prompts": prompts, 
                       "m_prompts": m_prompts, 
                       "v_hist_data": v_hist_data, 
                       "embeddings": embeddings, 
                       "v_hist_data": v_hist_data}
    with open(vector_file+".pkl", "wb") as f:
        pkl.dump(data_embeddings, f)

data_embeddings = pkl.load(open(vector_file+".pkl", "rb"))

prompts = data_embeddings["prompts"]
m_prompts = data_embeddings["m_prompts"]
v_hist_data = data_embeddings["v_hist_data"]


####################################################
##### APP        ###################################
####################################################

# chat history
chat_history = []

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['user_input']
        # Send user_input to ChatGPT and get the response
        response, computation_time, price = chat_with_gpt(user_input)
        # Add the input, output, computation time, and price to chat_history
        chat_history.append({
            'input': user_input,
            'output': response,
            'computation_time': computation_time,
            'price': price
        })
    return render_template('index.html', chat_history=chat_history)


def chat_with_gpt(user_input):

    # search in memory for similar prompts
    def search_in_memory(m_prompts, utterance):
        v_utterance = np.array(embeddings.embed_query(utterance))
        v_scores = m_prompts.dot(v_utterance)
        id, score = sorted(enumerate(v_scores.tolist()), key=lambda x: x[1], reverse=True)[0]
        return id, score
    
    id, score = search_in_memory(m_prompts, user_input)
    print("--log:: retrieven id: ", id, "score: ", score)

    # if agent is asked to generate a random tweet or "" is entered
    train_prompt = f"generate a random tweet from {twitter_handle}"
    if user_input == "" or user_input == train_prompt:
        user_input = train_prompt
        id = random.randint(0, len(v_hist_data))
    
    if score > similarity_threshold:
        response_text = v_hist_data[id]
        
        if search_mode == "exact" or run_mode == "light":
            return response_text, -1, -1
        
        elif search_mode == "mimic_response":
            if user_input != train_prompt:
                user_input = "generate a tweet quite similar to the following one:\n" + response_text

    # Send user_input to ChatGPT and get the response
    start_time = time.time()
    response = openai.Completion.create(
        engine=openai_engine,
        prompt=user_input,
        max_tokens=50,
        temperature=0.7,
        n = 1,
        stop=None,
    )
    end_time = time.time()
    computation_time = end_time - start_time
    price = computation_time * 0.000048  # Cost per second with text-davinci-003 engine

    response_text = response.choices[0].text.strip()
    return response_text, computation_time, price

if __name__ == '__main__':
    app.run(debug=True)

