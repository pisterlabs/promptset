import openai
import time
import pandas as pd
from glob import glob
from tqdm import tqdm

PROMPT = """Rewrite the following stock footage descriptors into complete sentences.  Be sure that the resulting sentences sound natural and remove specifics about cameras.

For example:
INPUT: 3d render of inky injections into water with luma matte. blue ink on white background 5
OUTPUT: Blue ink injections onto a white background.

INPUT: "Swimming in the pool ,slow motion 120 fps,handheld camera balanced steady shot " 
OUTPUT: A person swimming in the pool.

INPUT: Aerial drone isle of wight needles england travel sunrise
OUTPUT: The sun rises over the Isle of Wight Needles in England.

INPUT: CAPTION
OUTPUT: """

API_KEYS = [
    "example"
    # Begin with "sk-"
]

ENDPOINTS = [
    "gpt-3.5-turbo",
#    "gpt-3.5-turbo-16k",
#    "gpt-3.5-turbo-0613",
#    "gpt-3.5-turbo-16k-0613",
#    "gpt-3.5-turbo-0301" # Legacy
]

# logging info to choose optimal endpoint
end2stats = {key: {"total_time": 0, "num_succ_calls": 0, "num_fails": 0} for key in ENDPOINTS}

global endpoint_index
endpoint_index = 0

def get_next_endpoint():
    global endpoint_index
    endpoint_index += 1
    endpoint_index = endpoint_index % len(ENDPOINTS)
    return ENDPOINTS[endpoint_index]

global api_key_index
api_key_index = 0

def get_next_api_key():
    global api_key_index
    api_key_index += 1
    api_key_index = api_key_index % len(API_KEYS)
    return API_KEYS[api_key_index]

def get_response(prompt, endpoint):
    response = openai.ChatCompletion.create(
        model=endpoint,
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},  
                {"role": "user", "content": prompt},    
        ],
        temperature=0,
        max_tokens=100,
    )
    return response.choices[0]['message']["content"]


def get_prompt(caption):
    return PROMPT.replace("CAPTION", caption)

def prompt_chatgpt(caption, endpoint="gpt-3.5-turbo"):
    prompt = get_prompt(caption)
    response = get_response(prompt, endpoint)
    return response

openai.api_key = API_KEYS[0]
#print(prompt_chatgpt("Usd cash macro view. green 100 dollar cash."))

def get_caption(index):
    return texts[index]

def read_write_batch_entry(index):
    caption = get_caption(index)
    try: 
        openai.api_key = get_next_api_key()
        new_caption = prompt_chatgpt(caption)
    except:
        new_caption = "CHATGPT ERROR"
        #print("Error with caption", caption)
        #print("Was using api key: ", openai.api_key)
        #print("Switching to API key", openai.api_key)
        time.sleep(0.2)

    with open("batched_rewrites/output.txt", "a") as f:
        f.write(str(index)+ " " + new_caption + "\n")

# Main script
print("Reading from CSV...")
df = pd.read_csv("results_10M_train.csv")
texts = df['name'].tolist()
idxs = list(range(len(texts)))
start_idx = 2596000
idxs = idxs[start_idx:]
# Serial implementation
#for idx in tqdm(idxs):
#    read_write_batch_entry(idx)


import multiprocessing
from tqdm import tqdm

def parallel_processing(idxs, n_processes):
    with multiprocessing.Pool(processes=n_processes) as pool:
        # wrap the imap_unordered result with tqdm for progress display
        # since we know the length of idxs, we can set the total count for tqdm
        for _ in tqdm(pool.imap_unordered(read_write_batch_entry, idxs), total=len(idxs)):
            pass

# Usage:
print("Beginning parallel processing!")
n_processes = 12  # specify the number of processes you want to use
parallel_processing(idxs, n_processes)

"""
files = glob("rewrites_chatgpt/*.txt")
start_idx = 0
for file in files:
    with open(file, "r") as f:
        new_texts = f.readlines()
        start_idx += len(new_texts)

print("Starting from index", start_idx)

end_idx = 0
files = glob("rewrites_chatgpt/reverse_*.txt")
for file in files:
    with open(file, "r") as f:
        new_texts = f.readlines()
        end_idx += len(new_texts)

if end_idx == 0:
    texts = texts[start_idx:]
else:
    texts = texts[start_idx:-end_idx]

idx = start_idx

# Go in reverse order!
texts.reverse()

for caption in tqdm(texts):
    if idx % 100 == 0:
        print("Stats dump:", end2stats)
    num_consec_fails = 0
    success = False
    while not success:
        openai.api_key = get_next_api_key()
        endpoint = get_next_endpoint()
        try:
            #time.sleep(0.05)
            start = time.time()
            new_caption = prompt_chatgpt(caption, endpoint)
            end = time.time()
            end2stats[endpoint]["total_time"] += end - start
            end2stats[endpoint]["num_succ_calls"] += 1
            success = True
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print("Error", e)
            num_consec_fails += 1
            end2stats[endpoint]["num_fails"] += 1
            time.sleep(0.2)
            if num_consec_fails > 50:
                print("Too many consecutive fails, exiting")
                exit()
            if num_consec_fails > 6:
                time.sleep(1)
    idx += 1
    with open("rewrites_chatgpt/reverse_{}.txt".format(end_idx), "a") as f:
        f.write(new_caption + "\n")
"""