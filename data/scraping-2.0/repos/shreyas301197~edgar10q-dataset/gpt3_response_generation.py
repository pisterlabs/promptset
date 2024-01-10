import pandas as pd
import os
import openai
from tqdm.notebook import tqdm

list_of_files = os.listdir('./dataset')
temp = list_of_files[:188]

def get_response(file_name):    
    path_dir = './dataset/' + str(file_name)
    dump_dir = './GPT3Responses/' + str(file_name)
    df = pd.read_csv(path_dir,encoding='utf-8').iloc[:50]
    prompt= "Based on the example given below, generate entity-phrase pairs. \n\nSentence:\nIssuance of common stock in May 2019 public offering at $243.00 per share,  net of issuance costs of $15.\nEntity:\n$15\nPhrase:\nCommon stock public offering issuance costs\n\nGenerate Entity-phrase pairs based on the sentence below:\nSentence:\n"
    Entity_phrase ='\n\nEntity:\n'
    phrase = '\nPhrase: \n'
    sentences = df['sentence']
    entities = df['entity']
    ans_phrases = []
    for i in range (0,len(sentences)):  #change here
        final_prompt = prompt + sentences[i] + Entity_phrase + entities[i] + phrase
        openai.api_key = "  "  #add key here
        response = openai.Completion.create(
          model="text-davinci-002",
          prompt=final_prompt,
          temperature=0.7,
          max_tokens=693,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )

        ans_phrases.append(response['choices'][0]['text'])
    df['GPT3 Responses'] = ans_phrases
    df.to_csv(dump_dir)
    print("completed", file_name)

for i in tqdm(temp): 
    try:
        if ".csv" in i:
            get_response(i)
            
    except:
        print(i)
        continue