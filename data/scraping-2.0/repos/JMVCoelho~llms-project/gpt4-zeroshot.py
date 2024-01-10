import time
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import re


base = '/Users/aprameya/Desktop/llms-project/data/'
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4"
openai.api_key = ""


jsonObjdata = pd.read_json("/Users/aprameya/Downloads/corpus.jsonl", lines=True)
ans = []
count=0
d = dict()
for index, row in jsonObjdata.iterrows():
    d[row["doc_id"]] = row["page_title"]


 
jsonObj = pd.read_json(path_or_buf=base+"queries.jsonl", lines=True)
ans = []
actual = []
count=0
for index,row in jsonObj.iterrows():
    query = f"""You are an expert in movies. You are helping someone recollect a movie name that is on the tip of their tongue. You respond to each message with a single guess for the name of the movie being described.**important**: you only mention the names of the movie and nothing else. Given below is the movie description:"

        Description:
        \"\"\"
        {row["text"]}
        \"\"\"

        """
    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'system', 'content': 'You help someone recollect a movie'},
            {'role': 'user', 'content': query},
        ],
        model=GPT_MODEL,
        temperature=0,
    )
    ans.append(response['choices'][0]['message']['content'])
    actual.append(d[row['wikipedia_id']])
    print(count)
    count+=1
    time.sleep(3)
with open(base+'results.txt', 'w') as fp:
    for item in ans:
        fp.write(item+'\n')
with open(base+'labels.txt', 'w') as fp:
    for item in actual:
        fp.write(item+'\n')


### EVALUATE ###

with open(base+'results.txt', 'r') as f:
  guesses = f.readlines()
with open(base+'labels.txt', 'r') as f:
  labels = f.readlines()
count = 0
for i in range(len(labels)):
  labels[i] = re.sub("\(.*?\)|\[.*?\]","",labels[i])
  guesses[i] = re.sub("\(.*?\)|\[.*?\]","",guesses[i])
  if(labels[i].strip()==guesses[i].strip()):
    count+=1
    print(labels[i])

print(count/150) 

            