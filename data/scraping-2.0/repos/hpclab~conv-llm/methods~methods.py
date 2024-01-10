import pickle
import ast
import re
import os
import pandas as pd
def literal_return(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as e:
        print(val)
        return val

def convert_column_from_text_to_list(column:list):
    return [[re.sub('^\s+','',str(entity)) for entity in literal_return(str(x))] for x in column]


def write_dict(dictionary_data,output_path):
    with open(output_path,'wb') as file:
        pickle.dump(dictionary_data, file)

def load_dict(input_path):
    with open(input_path,"rb") as file:
        data = pickle.load(file)
    return data

def write_list(lista,outpath):
    with open(outpath, 'w') as f:
        for el in lista:
            f.write("%s\n" % el)
            
def read_list(path):
    with open(path, 'r') as f:
        lines= f.readlines()
        lines = [literal_return(x) for x in lines]
    return lines



### Takes an evaluation dataframe loaded with pd.read_json, elaborates the data and return a DataFrame ###

def create_df_from_json(evaluation):
    for i,el in enumerate(evaluation.turn):
        if i == 0:
            dictionary = {key : [] for key in el[0].keys()}
        for line in el:
            for key in line.keys():
                if key != 'number':
                    dictionary[key].append(line[key])
                else:
                    dictionary[key].append(str(evaluation.number.iloc[i])+'_'+str(line[key]))
    return pd.DataFrame.from_dict(dictionary)


### Takes the treccast path with .json files of the evaluation in the folder, load them all and return a DataFrame qith all the available years concatenated

def load_evaluation(treccast_path,load_train =True):
    if treccast_path[-1] != '/':  treccast_path += '/'
    if load_train:
        files = [treccast_path+name for name in os.listdir(treccast_path) if '.json' in name]
    else: 
        files = [treccast_path+name for name in os.listdir(treccast_path) if '.json' in name and 'train' not in name]
    evaluation = pd.DataFrame()
    for file in files:
        loaded =create_df_from_json(pd.read_json(file)).rename(columns={'number': 'qid'})[['qid', 'raw_utterance']]
        loaded.insert(1,'conv_id',[x.split('_')[0] for x in  loaded.qid.tolist()])
        loaded.insert(2,'turn',[x.split('_')[1] for x in  loaded.qid.tolist()])

        if '2019' in  file: loaded.insert(3,'year',[int(2019)] * len(loaded))
        elif '2020' in  file: loaded.insert(3,'year',[int(2020)] * len(loaded))
        elif '2021' in  file: loaded.insert(3,'year',[int(2021)] * len(loaded))
        elif '2022' in  file: loaded.insert(3,'year',[int(2022)] * len(loaded))
        evaluation = pd.concat([evaluation,loaded])
    #evaluation['raw_utterance'] = [terrier_query(x) for x in evaluation['raw_utterance'].tolist()]
    return evaluation


### Load a qrel file given the path and add a column with the year ###

def load_qrels(qrel_path='/trec-cast-qrels-docs.2021.qrel'):
    qrels = pd.read_csv(qrel_path, delimiter=" ", header=None)
    qrels[[3]] = qrels[[3]].astype(int)
    qrels = qrels.drop([1], axis=1)
    qrels.columns=["qid", "docno", "label"] 
    if '2019' in  qrel_path: qrels['year'] = [2019] * len(qrels)
    if '2020' in  qrel_path: qrels['year'] = [2020] * len(qrels)
    if '2021' in  qrel_path: qrels['year'] = [2021] * len(qrels)
    if '2022' in  qrel_path: qrels['year'] = [2022] * len(qrels)
    return qrels


### Load a qrel file given the path of the folder containing all the qrels it load them all and returns a DataFrame ###

def load_all_qrels(path_folder):
    if path_folder[-1] != '/': path_folder +='/'
    qrels = pd.DataFrame()
    paths =[path_folder+name for name in os.listdir(path_folder)]
    for name in paths:
        qrels = pd.concat([qrels,load_qrels(name) ])
    return qrels



import openai
import time
openai.api_key = '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'
def query_chatgpt(query):
    try:
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=query,
        temperature=0.6,
        max_tokens=150,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
        )
        return response.choices[0].text.replace('\n','')
    except:
        time.sleep(5)
        try:query_chatgpt(query)
        except:
            time.sleep(10)
            try:query_chatgpt(query)
            except:
                time.sleep(15)
                query_chatgpt(query)



import re
def sub_(x):
    
    #x= re.sub(r'.+:','',x)
    #x= re.sub(r'\n\n.+','',x)
    return x
