# %%
import pandas as pd
import numpy as np
import json, time, os, requests
import openai

PDB_URL_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
openai.api_key = "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"

# %%
def get_PDB_id_from_pharma_model(str):
    return str.split("_")[0].upper()

def get_ids_column(df):
    return df['Pharma Model'].map(get_PDB_id_from_pharma_model)


# %%
def get_keywords_column(df_ids):
    json_path = './keywords.json'
    if os.path.exists(json_path):
        with open(json_path,'r+') as json_file:
            keywords_dict = json.load(json_file)
    else:
        keywords_dict = {}
    #
    keywords_list = []
    data_len = len(df_ids)
    for x in range(data_len):
        if not (df_ids[x] in keywords_dict):
            entry_id = df_ids[x]
            URL_PDB_DATA = f"https://data.rcsb.org/rest/v1/core/entry/{entry_id}"
            response = requests.get(URL_PDB_DATA)
            if (response.status_code == 200):
                keywords_dict[df_ids[x]] = response.json()['struct_keywords']['pdbx_keywords']
                keywords_list.append(response.json()['struct_keywords']['pdbx_keywords'])
            else:
                keywords_dict[df_ids[x]] = "ERROR try this entry again"
                keywords_list.append("ERROR try this entry again")
        else:
            keywords_list.append(keywords_dict[df_ids[x]])
    #
    with open(json_path,'+w') as json_file:
        json.dump(keywords_dict, json_file)
    #
    return pd.Series(keywords_list, )

# %%
def refine_class(classification):
    if "OXIDOREDUCTASE" in classification:
        classification =  "OXIDOREDUCTASES"
    elif "TRANSFERASE" in classification:
        classification =  "TRANSFERASES"
    elif "HYDROLASE" in classification:
        classification =  "HYDROLASES"
    elif "LYASE" in classification:
        classification =  "LYASES"
    elif "ISOMERASE" in classification:
        classification =  "ISOMERASES"
    elif "LIGASE" in classification:
        classification =  "LIGASES"
    elif "TRANSLOCASE" in classification:
        classification =  "TRANSLOCASES"
    else:
        classification = "OTHER"
    return classification

def get_class_column(df, df_keywords):
    json_path = './classifications-gpt.json'
    if os.path.exists(json_path):
        with open(json_path,'r+') as json_file:
            class_dict_gpt = json.load(json_file)
    else:
        class_dict_gpt = {}
    #
    data_len = len(df_keywords)
    for x in range(data_len):
        if not (df_keywords[x] in class_dict_gpt):
            completion = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                temperature = 0.5,
                max_tokens = 300,
                messages = [
                    {"role": "system", "content": "You have to determine in which classification group fits best the given protein, by its name and a keyword associated with it. Which are given in name-keyword format"},
                    {"role": "system", "content": "Answer with only the classification no more words"} ,
                    {"role": "system", "content": "For each element of the list given you can ONLY choose from one of these options: oxidoreductases, transferases, hydrolases, lyases, isomerases, ligases, translocases, not an enzyme. Do not answer out of these options."},
                    {"role": "assistant", "content": "['oxidoreductase', 'not an enzyme', ...]"},
                    {"role": "user", "content": f"{df['Name'][x]},{df_keywords[x]}"}
                ]
            )
            class_dict_gpt[df_keywords[x]] = completion.choices[0].message.content.upper()
            time.sleep(20)
    #
    with open(json_path,'+w') as json_file:
        json.dump(class_dict_gpt, json_file)
    #
    refined_list = map(refine_class, class_dict_gpt.values())
    refined_class_dict_gpt = {key: refined_list for key, refined_list in zip(class_dict_gpt.keys(), refined_list)}
    #
    class_column = df_keywords.map(lambda keyword: refined_class_dict_gpt[keyword])
    return class_column


# %%
def get_trypanosomatida_column(df_ids):
    species = "Trypanosomatida"
    json = {
    "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                "operator": "exact_match",
                "value": f"{species}",
                "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name"
                }},
    "request_options": {
        "return_all_hits": True
    }
                ,
    "return_type": "entry"
    }
    response = requests.post(PDB_URL_SEARCH, json=json)
    results = response.json()['result_set']
    results_df = pd.DataFrame(results)
    return df_ids.isin(results_df['identifier'])

# %%
def organize_PHRMPR_data(data_path):
    '''Receives the PHARMMAPPER data search in CSV format, and returns a pandas Data Frame with the data
       organized following a certain crtiteria'''
    
    original_df = pd.read_csv(data_path,skiprows=1)
    result_df = pd.DataFrame()
    print("Creating Ids column")
    ids_column = get_ids_column(original_df)
    print("Creating Keywords column")
    keywords_column = get_keywords_column(ids_column)
    print("Creating Class column")
    class_column = get_class_column(original_df, keywords_column)
    print("Creating Trypanosomatida column")
    trypanosomatida_column = get_trypanosomatida_column(ids_column)

    result_df['Id'] = ids_column
    result_df['Fit'] = original_df['Fit']
    result_df['Norm Fit'] = original_df['Norm Fit']
    result_df['Name'] = original_df['Name']
    result_df['Keywords'] = keywords_column
    result_df['Class'] = class_column
    result_df['UNIPROT'] = original_df['Uniplot']
    result_df['Trypanosomatida'] = trypanosomatida_column

    print('Done')
    return result_df


# %%
test = organize_PHRMPR_data('./f9-300-conform1-PHARMMAPPER.csv')

