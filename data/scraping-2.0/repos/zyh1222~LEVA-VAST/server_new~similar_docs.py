import pickle
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
import openai
import json
from util import load_json,save_jl_append,save_txt,save_json
import os
openai.api_key = "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"

def format_time(d):
    b = "0" + str(d % 3600 // 60) if d % 3600 // 60 < 10 else str(d % 3600 // 60)
    c = "0" + str(d % 60) if d % 60 < 10 else str(d % 60)
    return str(d // 3600) + ":" + b + ":" + c

def process_and_save_data(input_path, output_path):
    data = pd.read_json(input_path)
    data = data[(data["label"] == 1.0) | (data["label"].isnull())]
    data["time"] = data["datetime"]
    data["datetime"] = data["datetime"].apply(format_time)
    data = data.set_index("datetime")
    data = data[['message', 'ID']]
    data['message'] = data['message'].astype(str)
    data['message'] = data['message'].str.lower()
    data.to_csv(output_path)

# Call the function to process and save the data
input_data_path = "../vast3/public/data.json"
output_data_path = './assets/data/data.csv'

VECTORSTORE_PATHS = {
    "geo": "./assets/data/vectorstore_geo.pkl",
    "all": "./assets/data/vectorstore.pkl"
}

DATA_PATH = {
    "geo": "./assets/data/geo_data.csv",
    "all": "./assets/data/data.csv"
}
def load_vectorstore(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def save_vectorstore(vectorstore, path):
    with open(path, 'wb') as file:
        pickle.dump(vectorstore, file)

def create_vectorstore(embeddings, data_name):
    path = VECTORSTORE_PATHS.get(data_name)
    if os.path.exists(path):
        print("Vectorstore loaded successfully.")
        return load_vectorstore(path)
    
    data_path = DATA_PATH.get(data_name)
    loader = CSVLoader(file_path=data_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    print(data)
    vectorstore = FAISS.from_documents(data, embeddings)
    print("Vectorstore created.")
    save_vectorstore(vectorstore, path)
    print("Vectorstore file saved.")
    return vectorstore

def get_similar_docs(question, data_name="all", df_search_context=None, k=10):
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    data_path = DATA_PATH.get(data_name)
    loaded_vectorstore = create_vectorstore(embeddings, data_name)
    docs = loaded_vectorstore.similarity_search_with_score(question, k)

    ids = [int(re.search(r'ID:\s*(\d+)', doc[0].page_content).group(1)) for doc in docs]
    print(ids)
    if df_search_context.empty:
        df_search_context = pd.read_csv(data_path)
    result = df_search_context[df_search_context['ID'].isin(ids)].drop_duplicates(subset=['message'])
    return result


def chat(text):
    
    completions = openai.ChatCompletion.create(
    model="all-3.5-turbo",
    messages=[
            {"role": "user", "content": text},
        ]
    )
    message = completions.choices[0].message
    return message["content"]

def get_response(keyword):

    save_path = f"./assets/data/{keyword}.txt"
    prompt_save_path = f"./assets/input/{keyword}_prompt.txt"
    if os.path.exists(save_path):
        with open(save_path, 'r') as file:
            context = file.read()
    else:
        question = """Some moron in a black van just hit my car !
FREE COFFEE to anyone who caught the license plate of that black van !
OMG ! ! ! ! ! ! ! Some derp in a ? balck ? van just ? hit a guy on a bike ! ! ! ! !
OMG ! A van just hit a guy on a bike .
Abila Police Department issues description of hit and run vehicle : a black van with partial license plate L829
has released description of hit and run vehicle from 500 block of Schaber : black van with partial license L829
wow some crazy black van just got pulled over in the parking lot - dinner & a show !
OMG dude in a black van is shooting at the cops ! Bullets everywhere .
Yeah van -- can't exit the backway at Gelato . And now the cops got him !
Better than expected - cops block van at Gelato . Moving to get a better view .
So van tried a t-turn to exit ; cop blocked him off ; van and cop doors open
Black van faces off two cop cars , shots fired , people undercover , not moving until its under control
Can't see who's in the van yet , one cop
why didnt i stay at the rally ? im trapped in here the van is right outside the door
cops and the van dude are hiding behind their doors - just like the movies
van guys behind their doors , cops too
Can't see if others in the van
officer has been shot by the driver of a black van
I wonder who this van dude is he looks like he wants to kill the cops
looks like there are 3 cop cars and the black van two people in the front of van
Wonder if the van guys are getting nervous or gonna give up
Police starting to move about - still can't see anyone in van - probably empty
van dude is waving his gun and shouting at cops . He's crazy !
guy in the van has a gun and he doesn't seem too stable
just realized the cops shot a lot of bullets and didn't hit the van dude
Guy just came back - says he looked out the back door and saw a black van and guys with guns .
Wanna get my van detailed - recommendations ?
guy in the van keeps yelling at cops cant hear him but looks mad
dude in the van says he's gonna shot a hostage if they dont let him go
black clothes guys yelling at van guys
reports shooter in black van has a hostage
I can hear the police yelling . sounds like the black van guys have captives .
Can't see any hostages in the van - they must be in the back
van guy yells back
I don't see anyone from the van . who said that ?
im just seeing someone talking to van guys . if there is anyone in the van they are still in there
they must have a van full of terrified people
swat guy is talking to van dude trying to get him to give up
this guy in the van is crazy why talk to that cant talk to crazy
Police have someone talking to van guys
Can't hear words - van guy sounds mad
they seemed really interested in where that van came from
It's a black panel van - no windows .
Negotiator guy and van guy still back and forth .
why has no one put it together ? black van shooter rally hostages = rally distraction for POK kidnapping POK
van guy waving gun at the roofs . Thinks SWATs there ?
evacuated witness reports seeing wild driving black van being pursued by two cop cars
van dude is yelling at cops
witness reports that male driver of black van tried to drive out of gelatogalore parking lot but was blocked then opened fire
guy in the black van is screaming something at the cops but cant hear what
i wonder why the cops were chasing this van anyway
think someone in the van just showed a hostage ! Looked like a girl
the other guy showed her through the van window
lots of guns pointed at that van
Kid says he saw 2 people in the van from the other side of the street
more talk - think van guys are trying to work a deal to leave
i wonder what the cops are offering the van dude to get him to give up
nothing new to report : heavily armed SWAT still in standoff with occupants of black van
van guy looking all around ?
what is this guy doing ? just give up and now hes back in the van
ACTION ! van guy bac in van
crazy guy is back in the van - what is going on ?
Moving van up for sale
Yelling FROM INSIDE the van !
the screaming guy is back in the van looks like hes in there screaming too - fighting with his partner maybe ?
looks like therye still fighting in the van really animated
someones out of the van ! is it over ?
other guy is out of the van ! its over theyre giving up !
Police are taking two van guys into waiting car
a woman is out of the passenger side van she has her hands up is it over ?
i think there giving up 2 people hands up got out of the van guy & girl
a man and woman have surrended to police two more people have left the van i think there hostages."""
        context = get_similar_docs(question,save_path, df_search_context = None, k=120)
    
    prompt = f"""Please summarize below messages data into a story of the "{keyword}" with several subevents. There may be messages that are not related to the story, so you may need to filter them. You can try to figure out when and where the subevent is. At the end of your answer, as each subevent may consist of one or more messages, you should use a list (e,g.,[1, 2,...]) to list the "ID" of the message. Use the messages as a source of facts, and do not engage in unfounded speculation. Subevent Title no more than 4 words. The output format:""" + """\n
    {
        "Story Title": short title,
        "Subevents": [
            {
            "Subevent": short title,
            "Location": "",
            "Time": "",
            "Messages": [],
            "Summary": ""
            }
        ]
    })""" + f"""\nDo not add other sentences.\n Data: \n{context}"""
    print(context)
    save_txt(prompt,prompt_save_path)

    res = chat(prompt)
    print(res)
    summary = json.loads(res)
    for i in range(len(summary['Subevents'])):
        int_list = list(map(int, summary['Subevents'][i]["Messages"]))
        summary['Subevents'][i]["Messages"] = int_list
    save_jl_append(summary, f'./assets/data/{keyword}_message_summary.json')
    json_response = json.dumps(summary)
    return json_response