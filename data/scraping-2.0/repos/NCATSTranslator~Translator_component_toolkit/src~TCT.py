import requests
import json
import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt
import ipycytoscape
import networkx as nx
import numpy as np
import openai


# used Dec 5, 2023 (Example_query_one_hop_with_category.ipynb)
def list_Translator_APIs():
    APInames = {
        #"BigGIM_BMG":"http://127.0.0.1:8000/find_path_by_predicate",
        "Aragorn(Trapi v1.4.0)":"https://aragorn.transltr.io/aragorn/query",
        "ARAX Translator Reasoner - TRAPI 1.4.0":"https://arax.transltr.io/api/arax/v1.4/asyncquery",
        "RTX KG2 - TRAPI 1.4.0":"https://arax.ncats.io/api/rtxkg2/v1.4/query",
        "SPOKE KP for TRAPI 1.4":"https://spokekp.transltr.io/api/v1.4/query",
        "Multiomics BigGIM-DrugResponse KP API":"https://bte.test.transltr.io/v1/smartapi/adf20dd6ff23dfe18e8e012bde686e31/query",
        "Multiomics ClinicalTrials KP":"https://api.bte.ncats.io/v1/smartapi/d86a24f6027ffe778f84ba10a7a1861a/query",
        "Multiomics Wellness KP API":"https://api.bte.ncats.io/v1/smartapi/02af7d098ab304e80d6f4806c3527027/query",
        "Multiomics EHR Risk KP API":"https://api.bte.ncats.io/v1/smartapi/d86a24f6027ffe778f84ba10a7a1861a/query",
        "Biothings Explorer (BTE)":"https://bte.transltr.io/v1/query",
        "Service Provider TRAPI":"https://api.bte.ncats.io/v1/smartapi/978fe380a147a8641caf72320862697b/query",
        "Explanatory-agent":"https://explanatory-agent-creative.azurewebsites.net/ARA/v1.3/asyncquery", #403 error
        "MolePro":"https://translator.broadinstitute.org/molepro/trapi/v1.4/asyncquery",
        "Genetics KP":"https://genetics-kp.transltr.io/genetics_provider/trapi/v1.4/query",
        "medikanren-unsecret":"https://medikanren-trapi.transltr.io/query",
        "Text Mined Cooccurrence API":"https://api.bte.ncats.io/v1/smartapi/978fe380a147a8641caf72320862697b/query",
        "OpenPredict API":"https://openpredict.transltr.io/query",
        "Agrkb(Trapi v1.4.0)":"https://automat.transltr.io/genome-alliance/1.4/query",
        "Automat-biolink(Trapi v1.4.0)": "https://automat.renci.org/biolink/1.4/query",
        "Automat-cam-kp(Trapi v1.4.0)": "https://automat.ci.transltr.io/cam-kp/1.4/query?limit=100",
        #"Automat-ctd(Trapi v1.4.0)": "https://automat.renci.org/drugcentral/1.4/query",
        "Automat-drug-central(Trapi v1.4.0)": "https://automat.ci.renci.org/drugcentral/1.4/query",
        "Automat-gtex(Trapi v1.4.0)":"https://automat.renci.org/gtex/1.4/query",
        "Automat-gtopdb(Trapi v1.4.0)": "https://automat.renci.org/gtopdb/1.4/query",
        "Automat-gwas-catalog(Trapi v1.4.0)": "https://automat.renci.org/gwas-catalog/1.4/query",
        "Automat-hetio(Trapi v1.4.0)": "https://automat.ci.transltr.io/hetio/1.4/query",
        "Automat-hgnc(Trapi v1.4.0)": "https://automat.renci.org/hgnc/1.4/query",
        "Automat-hmdb(Trapi v1.4.0)": "https://automat.renci.org/hmdb/1.4/query",
        "Automat-human-goa(Trapi v1.4.0)": "https://automat.renci.org/human-goa/1.4/query",
        "Automat-icees-kg(Trapi v1.4.0)": "https://automat.renci.org/icees-kg/1.4/query",
        "Automat-intact(Trapi v1.4.0)": "https://automat.renci.org/intact/1.4/query",
        "Automat-panther(Trapi v1.4.0)": "https://automat.renci.org/panther/1.4/query",
        "Automat-pharos(Trapi v1.4.0)": "https://automat.renci.org/pharos/1.4/query",
        "Automat-robokop(Trapi v1.4.0)": "https://ars-prod.transltr.io/ara-robokop/api/runquery", #doesn't work
        "Automat-sri-reference-kp(Trapi v1.4.0)": "https://automat.ci.transltr.io/sri-reference-kp/1.4/query", #doesn't work
        "Automat-string-db(Trapi v1.4.0)": "https://automat.ci.transltr.io/string-db/1.4/query",
        "Automat-ubergraph(Trapi v1.4.0)": "https://automat.ci.transltr.io/ubergraph/1.4/query",
        "Automat-ubergraph-nonredundant(Trapi v1.4.0)": "https://automat.ci.transltr.io/ubergraph-nonredundant/1.4/query",
        "Automat-viral-proteome(Trapi v1.4.0)": "https://automat.ci.transltr.io/viral-proteome/1.4/query",
        #"COHD TRAPI":"https://cohd-api.transltr.io/api/query", # 500 error
        "CTD API":"https://automat.ci.transltr.io/ctd/1.4/query",
        "Connections Hypothesis Provider API":"https://chp-api.transltr.io/query", #no knowledge_graph is defined in the response
        "MyGene.info API":"https://api.bte.ncats.io/v1/smartapi/59dce17363dce279d389100834e43648/query",
        "MyDisease.info API":"https://api.bte.ncats.io/v1/smartapi/671b45c0301c8624abbd26ae78449ca2/query",
        "MyChem.info API":"https://api.bte.ncats.io/v1/8f08d1446e0bb9c2b323713ce83e2bd3/query",
        "MyVariant.info API":"https://api.bte.ncats.io/v1/59dce17363dce279d389100834e43648/query",
        "Ontology Lookup Service API":"https://api.bte.ncats.io/v1/1c056ffc7ed0dd1229e71c4752239465/query",
        "PharmGKB REST API":"https://api.bte.ncats.io/v1/bde72db681ec0b8f9eeb67bb6b8dd72c/query",
        "QuickGO API":"https://api.bte.ncats.io/v1/1f277e1563fcfd124bfae2cc3c4bcdec/query",#pathways
        #"RaMP API v1.0.1":"",
        #"Text Mining Targeted Association API":"",
        "BioThings BindingDB API":"https://api.bte.ncats.io/v1/smartapi/38e9e5169a72aee3659c9ddba956790d/query",
        "BioThings BioPlanet Pathway-Disease API":"https://api.bte.ncats.io/v1/smartapi/55a223c6c6e0291dbd05f2faf27d16f4/query",
        "BioThings DDinter API":"https://api.bte.ncats.io/v1/smartapi/00fb85fc776279163199e6c50f6ddfc6/query",
        "BioThings DGIdb API":"https://api.bte.ncats.io/v1/smartapi/e3edd325c76f2992a111b43a907a4870/query",
        "BioThings DISEASES API":"https://api.bte.ncats.io/v1/smartapi/a7f784626a426d054885a5f33f17d3f8/query",
        "BioThings EBIgene2phenotype API":"https://api.bte.ncats.io/v1/smartapi/1f47552dabd67351d4c625adb0a10d00/query",
        "BioThings Biological Process API":"https://api.bte.ncats.io/v1/smartapi/cc857d5b7c8b7609b5bbb38ff990bfff/query",
        "BioThings GO Cellular Component API":"https://api.bte.ncats.io/v1/smartapi/f339b28426e7bf72028f60feefcd7465/query",
        "BioThings GO Molecular Function API":"https://api.bte.ncats.io/v1/smartapi/34bad236d77bea0a0ee6c6cba5be54a6/query",
        "BioThings GTRx API":"https://api.bte.ncats.io/v1/smartapi/316eab811fd9ef1097df98bcaa9f7361/query",
        "BioThings HPO API": "https://api.bte.ncats.io/v1/smartapi/d7d1cc9bbe04ad9936076ca5aea904fe/query",
        "BioThings IDISK API":"https://api.bte.ncats.io/v1/smartapi/32f36164fabed5d3abe6c2fd899c9418/query",
        "BioThings MGIgene2phenotype API":"https://api.bte.ncats.io/v1/smartapi/77ed27f111262d0289ed4f4071faa619/query",
        "BioThings PFOCR API":"https://api.bte.ncats.io/v1/smartapi/edeb26858bd27d0322af93e7a9e08761/query",
        "Biothings RARe-SOURCE API":"https://api.bte.ncats.io/v1/smartapi/b772ebfbfa536bba37764d7fddb11d6f/query",
        "BioThings Rhea API":"https://api.bte.ncats.io/v1/smartapi/03283cc2b21c077be6794e1704b1d230/query",
        "BioThings SEMMEDDB API":"https://api.bte.ncats.io/v1/smartapi/1d288b3a3caf75d541ffaae3aab386c8/query",
        "BioThings SuppKG API":"https://api.bte.ncats.io/v1/smartapi/b48c34df08d16311e3bca06b135b828d/query",
        "BioThings UBERON API":"https://api.bte.ncats.io/v1/smartapi/ec6d76016ef40f284359d17fbf78df20/query",
    }
    return(APInames)

# used Dec 5, 2023
def find_link(name):
    pre = "https://dev.smart-api.info/api/metakg/consolidated?size=2000&q=%28api.x-translator.component%3AKP+AND+api.name%3A"
    end = "%5C%28Trapi+v1.4.0%5C%29%29"
    if '(Trapi v1.4.0)' in name:
        url = pre
        name_raw = name.split("(")[0]
        words = name_raw.split(" ")
    
        length = len(words)
        if length == 1:
            url = url + words[0] + end
        else:
            for i in range(0,length-1):
                url = url + words[i] + "+"
            url = url+words[length-1]+end
    
    else:
        words = name.split(" ")
        url = pre
        length = len(words)
        
        for i in range(0,length-1):
            url = url + words[i] + "+"
        url = url+words[length-1]+"%29"
    return(url)

# used
# Finalized version: Dec 5, 2023 (Example_query_one_hop_with_category.ipynb)
def get_KP_metadata(APInames):

    '''
    APInames = get_API_list()
    '''

    result_df = pd.DataFrame()
    API_list = []
    URL_list = []
    KG_category_list = []
    subject_list = []
    object_list = []
    url_list = []
    #for KP in KPnames:
    for KP in APInames.keys():
        json_text ={}
        if KP == "RTX KG2 - TRAPI 1.4.0": 
            #print("ARAX KG2 - TRAPI 1.4.0")
            text =requests.get("https://dev.smart-api.info/api/metakg/consolidated?size=20&q=%28api.x-translator.component%3AKP+AND+api.name%3ARTX+KG2+%5C-+TRAPI+1%5C.4%5C.0%29").text
            json_text = json.loads(text)
        else:
            text = requests.get(find_link(KP)).text
            #text = requests.get(dic_KP_metadata[KP]).text
            json_text = json.loads(text)

        for i in (json_text['hits']):
            KG_category_list.append("biolink:"+i['_id'].split("-")[1])
            API_list.append(KP)
            subject_list.append('biolink:'+i['_id'].split("-")[0])
            object_list.append('biolink:'+i['_id'].split("-")[2])
            url_list.append(APInames[KP])

    result_df = pd.DataFrame({ 'API': API_list, 'KG_category': KG_category_list, "Subject":subject_list, "Object":object_list, "URL":url_list})
    #result_df.to_csv("KP_metadata.csv", index=False)
    return(result_df)

# used
def add_new_API_for_query(APInames, metaKG, newAPIname, newAPIurl, newAPIcategory, newAPIsubject, newAPIobject):

    '''
    This function is used to add a new API beyond the current list of APIs for query
    Example: APInames, metaKG = add_new_API_for_query(APInames, metaKG, "BigGIM_BMG", "http://127.0.0.1:8000/find_path_by_predicate", "Gene-physically_interacts_with-gene", "Gene", "Gene")

    '''
    APInames[newAPIname] = newAPIurl

    new_row = pd.DataFrame({"API":newAPIname,
                            "KG_category":newAPIcategory,
                            "Subject":newAPIsubject, "Object":newAPIobject,
                            "URL":newAPIurl}, index=[0])
    metaKG = pd.concat([metaKG, new_row], ignore_index=True)
    return APInames, metaKG

# used. Dec 5, 2023 (Example_query_one_hop_with_category.ipynb)
def select_API(sub_list,obj_list, metaKG):
    '''
    sub_list = ["biolink:Gene", "biolink:Protein"]
    obj_list = ["biolink:Gene", "biolink:Disease"]
    
    '''
    new_sub_list = sub_list
    new_obj_list = obj_list
    #for item in sub_list:
    #    new_sub_list.append(item.split(":")[1])
    #for item in obj_list:
    #    new_obj_list.append(item.split(":")[1])

    #metaKG = pd.read_csv("KP_metadata.csv")
    df1 = metaKG.loc[(metaKG['Subject'].isin(new_sub_list)) & (metaKG['Object'].isin(new_obj_list))]
    df2 = metaKG.loc[(metaKG['Subject'].isin(new_obj_list)) & (metaKG['Object'].isin(new_sub_list))]
    df = pd.concat([df1,df2])
    return(list(set(df['API'].values)))

# used. Dec 5, 2023  (Example_query_one_hop_with_category.ipynb)
def select_concept(sub_list,obj_list,metaKG):
    #result_df = pd.read_csv("KP_metadata.csv")
    df1 = metaKG.loc[(metaKG['Subject'].isin(sub_list)) & (metaKG['Object'].isin(obj_list))]
    df2 = metaKG.loc[(metaKG['Subject'].isin(obj_list)) & (metaKG['Object'].isin(sub_list))]
    df = pd.concat([df1,df2])
    return(set(list(df['KG_category'])))

# used. Dec 5, 2023 (Example_query_one_hop_with_category.ipynb)
def get_Translator_API_URL(API_sele, APInames):
    API_URL = []

    for name in API_sele:
        if name in APInames.keys():
            API_URL.append(APInames[name])
        else:
            print("API name not found")
    return API_URL

# select APIs based on the predicates. Dec 10, 2023
def filter_APIs(sele_predicates, metaKG):
    if sele_predicates == []:
        sele_API_URL = list(metaKG['KG_category'].unique())    
    else:
        sele_API_URL = list(metaKG.loc[metaKG['KG_category'].isin(sele_predicates)]['URL'].unique())
    return sele_API_URL

def select_predicates_inKP(sub_list,obj_list,KPname,metaKG):
    '''sub_list = ["biolink:Gene", "biolink:Protein"]
      obj_list = ["biolink:Gene", "biolink:Disease"]
      KPname = "" # it should be one of the names in APInames
    '''

    new_sub_list = []
    new_obj_list = []
    for item in sub_list:
        new_sub_list.append(item.split(":")[1])
    for item in obj_list:
        new_obj_list.append(item.split(":")[1])

    #result_df = pd.read_csv("KP_metadata.csv")
    df1 = metaKG.loc[(metaKG['Subject'].isin(new_sub_list)) & (metaKG['Object'].isin(new_obj_list)) & (metaKG['API']==KPname)]
    df2 = metaKG.loc[(metaKG['Subject'].isin(new_obj_list)) & (metaKG['Object'].isin(new_sub_list)) & (metaKG['API']==KPname)]
    df = pd.concat([df1,df2])
    temp_set = (set(list(df['KG_category'])))
    final_set = []
    for concept in temp_set:
        #final_set.append("biolink:"+concept.split("-")[1])
        final_set.append(concept)
    return(final_set)

    

# used. Dec 5, 2023  (Example_query_one_hop_with_category.ipynb)
def Gene_id_converter(id_list, API_url):
    id_list_new = []
    for id in id_list:
        if id.startswith("NCBIGene:"):
            id = id.replace("NCBIGene:", "NCBIGene")
            id_list_new.append(id)
    query_json = {
                    "message": {
                        "query_graph": {
                        "nodes": {
                            "n0": {
                            "categories": ["Gene"],
                            "ids": id_list_new
                            },
                            "n1": {
                            "categories": [
                                "string"
                            ],
                            "ids": [
                                "string"
                            ]
                            }
                        },
                        "edges": {
                            "e1": {
                            "predicates": [
                                "string"
                            ]
                            }
                        }
                        }
                    }
                    }

    response = requests.post(API_url, json=query_json)
    result = {}

    if response.status_code == 200:
        result = response.json()

    return(result)

# used. Dec 5, 2023 (Example_query_one_hop_with_category.ipynb)
def format_query_json(subject_ids, object_ids, subject_categories, object_categories, predicates):
    '''
    Example input:
    subject_ids = ["NCBIGene:3845"]
    object_ids = []
    subject_categories = ["biolink:Gene"]
    object_categories = ["biolink:Gene"]
    predicates = ["biolink:positively_correlated_with", "biolink:physically_interacts_with"]

    '''
    #edited Dec 5, 2023
    query_json_temp = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {
                        "ids":[],
                        "categories":[]
                    },
                    "n1": {
                        #"ids":[],
                        "categories":[]
                }
                },
                "edges": {
                    "e1": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": predicates
                    }
                }
            }
        }
    }

    if len(subject_ids) > 0:
        query_json_temp["message"]["query_graph"]["nodes"]["n0"]["ids"] = subject_ids
    if len(object_ids) > 0:
        query_json_temp["message"]["query_graph"]["nodes"]["n1"]["ids"] = object_ids

    if len(subject_categories) > 0:
        query_json_temp["message"]["query_graph"]["nodes"]["n0"]["categories"] = subject_categories

    if len(object_categories) > 0:
        query_json_temp["message"]["query_graph"]["nodes"]["n1"]["categories"] = object_categories

    if len(predicates) > 0:
        query_json_temp["message"]["query_graph"]["edges"]["e1"]["predicates"] = predicates

    return(query_json_temp)

# used. Dec 5, 2023 (Example_query_one_hop_with_category.ipynb)
def query_KP(remoteURL, query_json):
    # Single query
    response = requests.post(remoteURL, json=query_json)
    #print(response.status_code)
    if response.status_code == 200:
        #print(response.json())[0]
        if "message" in response.json():
            result = response.json()["message"]
            #print(result) # revised Dec 1, 2023
            if "knowledge_graph" in result:
                #return(result['knowledge_graph'])
                return(result)  
            else:
                return()
        else:
            return()
    else:
        print("Error: " + str(response.status_code))
        return()
    
# used. Dec 5, 2023 (Example_query_one_hop_with_category.ipynb)
def parallel_api_query(URLS, query_json, max_workers=1):
    # Parallel query
    result = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(query_KP, url, query_json): url for url in URLS}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                if 'knowledge_graph' in data:
                    result.append(data)
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))
    
    included_KP_ID = []
    for i in range(0,len(result)):
        if result[i]['knowledge_graph'] is not None:
            if 'knowledge_graph' in result[i]:
                if 'edges' in result[i]['knowledge_graph']:
                    if len(result[i]['knowledge_graph']['edges']) > 0:
                        included_KP_ID.append(i)

    result_merged = {}
    for i in included_KP_ID:
        result_merged = {**result_merged, **result[i]['knowledge_graph']['edges']}

    len(result_merged)

    return(result_merged)

# used. Dec 5, 2023    (Example_query_one_hop_with_category.ipynb)
def parse_KG(result):
    '''
    subject_object
    subject
    object
    predicate
    primary_knowledge_sources
    aggregator_knowledge_sources
    subject_predicate_object_primary_knowledge_sources_aggregator_knowledge_sources
    
    '''
    # edited Dec 5, 2023

    result_parsed = {}
    for i in result:
        
        subject_object = result[i]['subject'] + "_" + result[i]['object']
        object_subject = result[i]['object'] + "_" + result[i]['subject']
        #result_parsed["predicate"].append(result[i]['predicate'])
        #result_parsed["sources"].append(result[i]['sources'])
        #result_parsed["subject"].append(result[i]['subject'])
        #result_parsed["object"].append(result[i]['object'])
        if subject_object not in result_parsed:
            result_parsed[subject_object] = {}
            result_parsed[subject_object]['predicate'] = [result[i]['predicate']]
            result_parsed[subject_object]['subject'] = result[i]['subject']
            result_parsed[subject_object]['object'] = result[i]['object']
            
            
            for j in result[i]['sources']:
                if j['resource_role'] == 'primary_knowledge_source':
                    result_parsed[subject_object]['primary_knowledge_source'] = [j['resource_id']]

                evidence =  result[i]['subject'] + "_" + result[i]['predicate'] + "_" + result[i]['object'] + "_" + j['resource_id']

                if j['resource_role'] == 'aggregator_knowledge_source':
                    result_parsed[subject_object]['aggregator_knowledge_source'] = [j['resource_id']]
                    evidence = evidence + "_" + j['resource_id']
            result_parsed[subject_object]['evidence'] = [evidence]

        else: # subject_object in result_parsed:
            result_parsed[subject_object]['predicate'].append(result[i]['predicate'])
            for j in result[i]['sources']:
                if j['resource_role'] == 'primary_knowledge_source':
                    result_parsed[subject_object]['primary_knowledge_source'].append(j['resource_id'])
                    evidence =  result[i]['subject'] + "_" + result[i]['predicate'] + "_" + result[i]['object'] + "_" + j['resource_id']
                if j['resource_role'] == 'aggregator_knowledge_source':
                    if 'aggregator_knowledge_source' not in result_parsed[subject_object]:
                        result_parsed[subject_object]['aggregator_knowledge_source'] = [j['resource_id']]
                    else:
                        result_parsed[subject_object]['aggregator_knowledge_source'].append(j['resource_id'])
                    evidence = evidence + "_" + j['resource_id']
            result_parsed[subject_object]['evidence'].append(evidence)

    return(result_parsed)


# parse network results. Dec 10, 2023
def parse_network_result(result, input_node1_list):
    dic_nodes = {}
    for i in result:
        subject = result[i]['subject']
        object = result[i]['object']
        predicate = result[i]['predicate']
        sources = result[i]['sources']

        if subject == object:
            continue
        
        if subject in dic_nodes:
            dic_nodes[subject].append(object)
        else:
            dic_nodes[subject] = [object]
        
        if object in dic_nodes:
            dic_nodes[object].append(subject)
        else:
            dic_nodes[object] = [subject]



    dic_remain_nodes = {}

    dic_with_input_nodes = {}

    for i in dic_nodes:
        if i in input_node1_list:
            dic_remain_nodes[i] = dic_nodes[i]
        else:
            continue

    for i in dic_remain_nodes:
        for j in dic_nodes[i]:
            if j in dic_with_input_nodes:
                dic_with_input_nodes[j].append(i)
            else:
                dic_with_input_nodes[j] = [i]

    for i in dic_with_input_nodes:
        dic_with_input_nodes[i] = list(set(dic_with_input_nodes[i]))
            



    for i in dic_with_input_nodes:
        if len(set(dic_with_input_nodes[i])) > 1: #
            #print(i, set(dic_with_input_nodes[i]))
            if i not in dic_remain_nodes:
                dic_remain_nodes[i] = dic_with_input_nodes[i]
        else:
            continue

    dic_remain_nodes_final = {}
    for i in dic_remain_nodes:
        dic_remain_nodes_final[i] = set(dic_remain_nodes[i]).intersection(set(dic_remain_nodes.keys()))


    subject_nodes = []
    object_nodes = []

    for i in dic_remain_nodes_final:
        for j in dic_remain_nodes_final[i]:
            subject_nodes.append(i)
            object_nodes.append(j)

    result_df = pd.DataFrame({'Subject':subject_nodes, 'Object':object_nodes})
    #result_df.to_csv('result_df.csv', index=False)
    return result_df

# parse results to a dictionary. Dec 5, 2023
# used. Dec 5, 2023 (Example_query_one_hop_with_category.ipynb)
def rank_by_primary_infores(result_parsed, input_node):
    ''' Editd Dec 5, 2023'''
    rank_df = pd.DataFrame()
    output_nodes = []
    Num_of_primary_infores = []
    type_of_nodes   = []
    for i in result_parsed:
        subject = result_parsed[i]['subject']
        object = result_parsed[i]['object']

        if subject == input_node:
            output_nodes.append(object)
            type_of_nodes.append('object')
            Num_of_primary_infores.append(len(set(result_parsed[i]['primary_knowledge_source'])))
        elif object == input_node:
            output_nodes.append(subject)
            type_of_nodes.append('subject')
        
            Num_of_primary_infores.append(len(set(result_parsed[i]['primary_knowledge_source'])))

    rank_df['output_node'] = output_nodes
    rank_df['Num_of_primary_infores'] = Num_of_primary_infores
    rank_df['type_of_nodes'] = type_of_nodes
    
    
    rank_df_ranked = rank_df.sort_values(by=['Num_of_primary_infores'], ascending=False)
    return(rank_df_ranked)

# used. Dec 5, 2023 (Example_query_one_hop_with_category.ipynb)
def visulization_one_hop_ranking(result_ranked_by_primary_infores,result_parsed , num_of_nodes = 50, input_query = "NCBIGene:3845"):
    # edited Dec 5, 2023
    predicates_list = []
    primary_infore_list = []
    aggregator_infore_list = []


    for i in range(0, result_ranked_by_primary_infores.shape[0]):
        oupput_node = result_ranked_by_primary_infores['output_node'][i]
        type_of_node = result_ranked_by_primary_infores['type_of_nodes'][i]
        if type_of_node == 'object':
            subject = input_query
            object = oupput_node
        else:
            subject = oupput_node
            object = input_query
            
        predicates_list = predicates_list + result_parsed[subject + "_" + object]['predicate']
        primary_infore_list = primary_infore_list + result_parsed[subject + "_" + object]['primary_knowledge_source']
        
        if 'aggregator_knowledge_source' in result_parsed[subject + "_" + object]:
            aggregator_infore_list = aggregator_infore_list + result_parsed[subject + "_" + object]['aggregator_knowledge_source']
            aggregator_infore_list = list(set(aggregator_infore_list))

        predicates_list = list(set(predicates_list))
        primary_infore_list = list(set(primary_infore_list))
        

    predicates_by_nodes = {}
    for predict in predicates_list:
        predicates_by_nodes[predict] = []

    primary_infore_by_nodes = {}
    for predict in primary_infore_list:
        primary_infore_by_nodes[predict] = []

    aggregator_infore_by_nodes = {}
    for predict in aggregator_infore_list:
        aggregator_infore_by_nodes[predict] = []
        
    names = []
    for i in range(0, result_ranked_by_primary_infores.shape[0]):
    #for i in range(0, 10):
        oupput_node = result_ranked_by_primary_infores['output_node'].values[i]
        names.append(oupput_node)
        type_of_node = result_ranked_by_primary_infores['type_of_nodes'].values[i]
        if type_of_node == 'object':
            subject = input_query
            object = oupput_node
        else:
            subject = oupput_node
            object = input_query
        new_id = subject + "_" + object

        cur_primary_infore = result_parsed[new_id]['primary_knowledge_source']
        for predict in primary_infore_list:
            if predict in cur_primary_infore:
                primary_infore_by_nodes[predict].append(1)
            else:
                primary_infore_by_nodes[predict].append(0)



        cur_predicates = result_parsed[new_id]['predicate']
        for predict in predicates_list:
            if predict in cur_predicates:
                predicates_by_nodes[predict].append(1)
            else:
                predicates_by_nodes[predict].append(0)

    convert = False
    colnames = names
    for item in colnames:
        if 'NCBIGene' in item:
            convert = True
    if convert:
        Gene_id_map = Gene_id_converter(colnames, "http://127.0.0.1:8000/query_name_by_id")
    
        new_colnames = []
        for item in colnames:
            if item in Gene_id_map.keys():
                new_colnames.append(Gene_id_map[item])
            else:
                new_colnames.append(item)    

    else:
        new_colnames = colnames
    primary_infore_by_nodes_df = pd.DataFrame(primary_infore_by_nodes)
    primary_infore_by_nodes_df.index = new_colnames
    primary_infore_by_nodes_df = primary_infore_by_nodes_df.T


    predicates_by_nodes_df = pd.DataFrame(predicates_by_nodes)
    predicates_by_nodes_df.index = new_colnames
    predicates_by_nodes_df = predicates_by_nodes_df.T

    title = "Ranking of one-hop nodes by primary infores"
    ylab = "infores"
    df = primary_infore_by_nodes_df.iloc[:,0:num_of_nodes]
    plt.figure( figsize=(0.8+df.shape[1]*0.2,3.5),dpi = 300)
        #p1 = sns.heatmap(df, cmap="Blues")
        # heatmap without color bar
    p1 = sns.heatmap(df, cmap="Blues", cbar=False)
    p1.set_title(title)
    p1.set_ylabel(ylab)
        # set title font size
    p1.title.set_size(12)

    title = "Ranking of one-hop nodes by predicate"
    ylab = "Predicate"
    df = predicates_by_nodes_df.iloc[:,0:num_of_nodes]
    plt.figure( figsize=(0.8+df.shape[1]*0.2,3.5),dpi = 300)
        #p1 = sns.heatmap(df, cmap="Blues")
        # heatmap without color bar
    p1 = sns.heatmap(df, cmap="Blues", cbar=False)
    p1.set_title(title)
    p1.set_ylabel(ylab)
        # set title font size
    p1.title.set_size(12)


    return(p1)
 

# used. Dec 5, 2023 (Example_query_rank_the_path.ipynb)
def merge_by_ranking_index(result_ranked_by_primary_infores,result_ranked_by_primary_infores2, top_n = 20):

    dic_rank1 = {}
    for i in range(0, result_ranked_by_primary_infores.shape[0]):
        dic_rank1[result_ranked_by_primary_infores['output_node'][i]] = 1 - i / result_ranked_by_primary_infores.shape[0]

    dic_rank2 = {}
    for i in range(0, result_ranked_by_primary_infores2.shape[0]):
        dic_rank2[result_ranked_by_primary_infores2['output_node'][i]] = 1 - i / result_ranked_by_primary_infores2.shape[0]

    merged_nodes = set(dic_rank1.keys()).intersection(set(dic_rank2.keys()))
    dic_merged_rank = {}

    for node in merged_nodes:
        dic_merged_rank[node] = dic_rank1[node] * dic_rank2[node]

    result_ranked = pd.DataFrame.from_dict(dic_merged_rank, orient='index', columns=['score'])
    result_ranked = result_ranked.sort_values(by=['score'], ascending=False)
    result_ranked = result_ranked.reset_index()
    result_ranked.columns = ['output_node', 'score']
    result_xy_sorted = result_ranked
    result_xy_sorted.index = result_ranked['output_node']

    convert = False
    colnames = result_xy_sorted.index.to_list()

    for item in colnames:
        if 'NCBIGene' in item:
            convert = True
    if convert:
        Gene_id_map = Gene_id_converter(colnames, "http://127.0.0.1:8000/query_name_by_id")
            
        new_colnames = []
        for item in colnames:
            if item in Gene_id_map.keys():
                new_colnames.append(Gene_id_map[item])
            else:
                new_colnames.append(item)    

    else:
        new_colnames = colnames

    result_xy_sorted.index = new_colnames
    result_xy_sorted = result_xy_sorted.sort_values(by=['score'], ascending=False)

    sns.set(style="whitegrid")
    plt.figure(figsize=(5,5), dpi = 300)
    ax = sns.barplot(x=result_xy_sorted.iloc[0:top_n].index, y=result_xy_sorted.iloc[0:top_n]['score'], color='grey')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    ax.set_ylabel("Ranking score")
    plt.tight_layout()
    plt.show()

    return result_xy_sorted



def merge_ranking_by_number_of_infores(result_ranked_by_primary_infores, result_ranked_by_primary_infores1, top_n = 50):
    overlapped = (set(result_ranked_by_primary_infores1['output_node']).intersection(set(result_ranked_by_primary_infores['output_node'])))
    x = result_ranked_by_primary_infores.loc[result_ranked_by_primary_infores['output_node'].isin(overlapped)]
    y = result_ranked_by_primary_infores1.loc[result_ranked_by_primary_infores1['output_node'].isin(overlapped)]
    dic_x = {}
    for i in range(x.shape[0]):
        dic_x[x.iloc[i]['output_node']] = x.iloc[i]['Num_of_primary_infores']/np.max(x['Num_of_primary_infores'])

    dic_y = {}
    for i in range(y.shape[0]):
        dic_y[y.iloc[i]['output_node']] = y.iloc[i]['Num_of_primary_infores']/np.max(y['Num_of_primary_infores'])

    dic_xy = {}
    for i in overlapped:
        dic_xy[i] = dic_x[i] * dic_y[i]
        
    result_xy = pd.DataFrame.from_dict(dic_xy, orient='index', columns=['score'])

    result_xy_sorted = result_xy.sort_values(by=['score'], ascending=False)

    convert = False
    colnames = result_xy_sorted.index.to_list()
    for item in colnames:
        if 'NCBIGene' in item:
            convert = True
    print(convert)
    if convert:
        Gene_id_map = Gene_id_converter(colnames, "http://127.0.0.1:8000/query_name_by_id")
        
        new_colnames = []
        for item in colnames:
            if item in Gene_id_map.keys():
                new_colnames.append(Gene_id_map[item])
            else:
                new_colnames.append(item)    

    else:
        new_colnames = colnames

    result_xy_sorted.index = new_colnames


    # barplot of the top 50 genes using sns
    sns.set(style="whitegrid")
    plt.figure(figsize=(5,5), dpi = 300)
    ax = sns.barplot(x=result_xy_sorted.iloc[0:top_n].index, y=result_xy_sorted.iloc[0:top_n]['score'], color='grey')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    ax.set_ylabel("Ranking score")
    plt.tight_layout()
    plt.show()
    return result_xy_sorted

# Sri-name-resolver  Used Dec 5, 2023 (Example_query_one_hop_with_category.ipynb)
def get_curie(name):
    url = "https://name-lookup.ci.transltr.io/lookup?string="+name
    response = requests.get(url)
    if response.status_code == 200:
        result = response.json()
        if len(result) != 0:
            return(result[0]['curie'])
        else:
            return(name)
    else:
        return(name)


#used
def query_chatGPT(customized_input):
    message=[{"role": "user", 
            "content": customized_input}]

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=1000,
    temperature=1.2,
    messages = message)

    print(len(response.choices[0].message.content.split(" ")))

    return(response.choices[0].message.content)


# to be removed
def query_KP_all(subject_ids, object_ids, subject_categories, object_categories, predicates, API_list,metaKG, APInames):

    #APInames = API_list
    if len(API_list) == 0:
        API_list = select_API(subject_categories,object_categories,metaKG)
    else:
        API_list = list(APInames.keys())

    result_dict = {}
    result_concept = {}
    # Query individual KP
    
    # Needs parallel query
    

    for API_sele in API_list:
        print(API_sele)
        if len(predicates)==0:
            predicates_used = select_predicates_inKP(subject_categories,object_categories,API_sele,metaKG)
        else:
            predicates_used = predicates
        
        query_json = format_query_json(subject_ids, object_ids, subject_categories, object_categories, predicates_used)

        print(query_json)
        try:
            kg_output = query_KP(APInames[API_sele],query_json)
            
        except:
            print("Connection Error")
            kg_output = None
            
        if kg_output is not None:
            # if kg_output is  a dictionary

            if type(kg_output) == dict and 'nodes' in kg_output.keys():
                if len(kg_output['nodes']) >0:

                    print("Found: " + str(len(kg_output['edges'].keys())) + " nodes in " + API_sele) 
                    print(predicates_used)
                    result_concept[API_sele] = predicates_used
                    result_dict[API_sele] = kg_output
    return(result_dict, result_concept)

# to be removed
def format_query_json_old(subject_ids, object_ids, subject_categories, object_categories, predicates):
    '''
    Example input:
    subject_ids = ["NCBIGene:3845"]
    object_ids = []
    subject_categories = ["biolink:Gene"]
    object_categories = ["biolink:Gene"]
    predicates = ["biolink:positively_correlated_with", "biolink:physically_interacts_with"]

    '''
    query_json_temp = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {
                        "ids":[],
                        "categories":[]
                    },
                    "n1": {
                        "categories":[]
                }
                },
                "edges": {
                    "e1": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": predicates
                    }
                }
            }
        }
    }

    if len(subject_ids) > 0:
        query_json_temp["message"]["query_graph"]["nodes"]["n0"]["ids"] = subject_ids
    if len(object_ids) > 0:
        query_json_temp["message"]["query_graph"]["nodes"]["n1"]["ids"] = object_ids

    if len(subject_categories) > 0:
        query_json_temp["message"]["query_graph"]["nodes"]["n0"]["categories"] = subject_categories

    if len(object_categories) > 0:
        query_json_temp["message"]["query_graph"]["nodes"]["n1"]["categories"] = object_categories

    if len(predicates) > 0:
        query_json_temp["message"]["query_graph"]["edges"]["e1"]["predicates"] = predicates

    return(query_json_temp)

# to be removed
def parse_result_old( API_keys_sele, API_keys_Not_include, predicates_forAnalysis,result_dic):
    Temp_APIkey = []
    Temp_subject_key = []
    Temp_object_key = []
    Temp_predicate_key = []
    Temp_infores_key = []
    API_keys_forAnalysis = []

    ALL_APIs_in_result = list(result_dic.keys())
    print(ALL_APIs_in_result)

    if len(API_keys_sele) == 0:
        API_keys_forAnalysis = ALL_APIs_in_result
    else:
        API_keys_forAnalysis = list(set(ALL_APIs_in_result).intersection(set(API_keys_sele)))

    if len(API_keys_Not_include) != 0:
        API_keys_forAnalysis = list(set(API_keys_forAnalysis) - set(API_keys_Not_include))


    print(API_keys_forAnalysis)

    for API_key in API_keys_forAnalysis:
        cur_API_outputKeys = list(result_dic[API_key]['edges'].keys())
        for i in range(0, len(cur_API_outputKeys)):
            curr_key = i
            curr_graph = (result_dic[API_key]['edges'][cur_API_outputKeys[curr_key]])
            predicate = (curr_graph['predicate'])
            if predicate != "biolink:subclass_of":
                infores = (curr_graph['sources'][0]['resource_id'])
                subject = (curr_graph['subject'])

                if subject.startswith("CL:"):
                    subject = "CL" + subject.split(":")[1]

                object = (curr_graph['object'])
                if object.startswith("CL:"):
                    object = "CL" + object.split(":")[1]
                
                #exclude subclass_of
                
                Temp_APIkey.append(API_key)
                Temp_subject_key.append(subject)
                Temp_object_key.append(object)
                Temp_predicate_key.append(predicate)
                Temp_infores_key.append(infores)
            
            #Temp_APIkey.append(API_key)
            #Temp_subject_key.append(subject)
            #Temp_object_key.append(object)
            #Temp_predicate_key.append(predicate)
            #Temp_infores_key.append(infores)

    Temp_result_df = pd.DataFrame({'API': Temp_APIkey, 
                                   'Subject': Temp_subject_key,
                                   "Object":Temp_object_key, 
                                   "Predicate":Temp_predicate_key, 
                                   "Infores":Temp_infores_key})

    Temp_result_df.drop_duplicates(inplace=True)
    Temp_result_df = Temp_result_df.loc[Temp_result_df['API'].isin(API_keys_forAnalysis)]

    if len(predicates_forAnalysis) != 0:
        Temp_result_df = Temp_result_df.loc[Temp_result_df['Predicate'].isin(predicates_forAnalysis)]
    return(Temp_result_df)

# to be removed
def ranking_result_by_predicates_object(Temp_result_df):
    object_val_list = Temp_result_df['Object'].value_counts().index.tolist()
    object_val_value = Temp_result_df['Object'].value_counts().values.tolist()

    
    dic_rank = {}
    for i in range(0,len(object_val_list)):
        dic_rank[object_val_list[i]] = object_val_value[i]
  

    sorted_dic = sorted(dic_rank.items(), key=lambda x: x[1], reverse=True)
    return(sorted_dic)

# to be removed
def ranking_result_by_predicates_subject(Temp_result_df):
    subject_val_list = Temp_result_df['Subject'].value_counts().index.tolist()
    subject_val_list = Temp_result_df['Subject'].value_counts().values.tolist()

    dic_rank = {}
    for i in range(0,len(subject_val_list)):
        dic_rank[subject_val_list[i]] = subject_val_list[i]
  

    sorted_dic = sorted(dic_rank.items(), key=lambda x: x[1], reverse=True)
    return(sorted_dic)



# to be removed
def get_ranking_by_predicates(sorted_dic, Temp_result_df, Top):
    #item_ranking = []
    dic_ranking = {}

    if Top > len(sorted_dic):
        Top = len(sorted_dic)

    for i in range(1,Top):
        #item_ranking.append(sorted_dic[i][0])
        sele_result = sorted_dic[i][0]
        dic_ranking[sorted_dic[i][0]] = list(set(list(pd.concat([Temp_result_df.loc[Temp_result_df['Object'].isin([sele_result])], Temp_result_df.loc[Temp_result_df['Subject'].isin([sele_result])]], axis=0)['Predicate'])))
    
    return(dic_ranking)

# to be removed
def get_ranking_by_infores(sorted_dic, Temp_result_df, Top):
    #item_ranking = []
    dic_ranking = {}

    if Top > len(sorted_dic):
        Top = len(sorted_dic)

    for i in range(1,Top):
        #item_ranking.append(sorted_dic[i][0])
        sele_result = sorted_dic[i][0]
        dic_ranking[sorted_dic[i][0]] = list(set(list(pd.concat([Temp_result_df.loc[Temp_result_df['Object'].isin([sele_result])], Temp_result_df.loc[Temp_result_df['Subject'].isin([sele_result])]], axis=0)['Infores'])))
    
    return(dic_ranking)

# to be removed
def get_ranking_by_kp(sorted_dic, Temp_result_df, Top):
    #item_ranking = []
    dic_ranking = {}

    if Top > len(sorted_dic):
        Top = len(sorted_dic)

    for i in range(1,Top):
        #item_ranking.append(sorted_dic[i][0])
        sele_result = sorted_dic[i][0]
        dic_ranking[sorted_dic[i][0]] = list(set(list(pd.concat([Temp_result_df.loc[Temp_result_df['Object'].isin([sele_result])], Temp_result_df.loc[Temp_result_df['Subject'].isin([sele_result])]], axis=0)['API'])))
    
    return(dic_ranking)

# to be revised
def connecting_two_dots_two_hops(sorted_dic1, sorted_dic):
    intermediate = []
    normalized_rank = []

    rank1 = 0
    for i in sorted_dic1:
        gene1 = i[0]

        rank1 = rank1 + 1
        rank2 = 0
        for j in sorted_dic:
            gene2 = j[0]
            rank2 = rank2 + 1
            if gene1 == gene2:
                normlized_rank1 = rank1/(len(sorted_dic1) -1)
                normlized_rank2 = rank2/(len(sorted_dic) -1)
                new_order = normlized_rank1 * normlized_rank2
                intermediate.append(gene2)
                normalized_rank.append(new_order)

    res_df = pd.DataFrame({"node":intermediate, "normalized_rank":normalized_rank})
    res_df.sort_values(by=['normalized_rank'], inplace=True, ascending=True)
    res_df.reset_index(inplace=True, drop=True)

    return(res_df)

def select_result_to_analysis(sele_genes,Temp_result_df, Temp_result_df1 ):
    
    print(sele_genes)
    for_plot = pd.concat([Temp_result_df1.loc[Temp_result_df1['Object'].isin(sele_genes)],
            Temp_result_df.loc[Temp_result_df['Object'].isin(sele_genes)]], axis=0)

    return(for_plot)

# need revision
def find_path_by_two_ends(subject1_ids, 
                          subject1_categories, 
                          predicates1,
                          object_categories,
                          subject2_ids, 
                          subject2_categories,
                          predicates2,
                          API_list1,
                          API_list2,
                          API1_keys_forAnalysis,
                          API1_keys_NotforAnalysis,
                          API2_keys_forAnalysis,
                          API2_keys_NotforAnalysis,
                          metaKG,
                          APInames
                          ):
    
    result_dic_node1, result_concept_node1 = query_KP_all(subject1_ids, [], subject1_categories, object_categories, predicates1, API_list1, metaKG, APInames)
    result_dic_node2, result_concept_node2 = query_KP_all(subject2_ids, [], subject2_categories, object_categories, predicates2, API_list2, metaKG, APInames)

    
    Temp_result_df1 = parse_result(API1_keys_forAnalysis,API1_keys_NotforAnalysis, result_concept_node1, result_dic_node1)
    sorted_dic1 = ranking_result_by_predicates_object(Temp_result_df1)

    dic_ranking1 = get_ranking_by_infores(sorted_dic1, Temp_result_df1, 20)

    Temp_result_df2 = parse_result(API2_keys_forAnalysis,API2_keys_NotforAnalysis, result_concept_node2, result_dic_node2)
    sorted_dic2 = ranking_result_by_predicates_object(Temp_result_df2)

    dic_ranking2 = get_ranking_by_infores(sorted_dic2, Temp_result_df2, 20)
    
    connection_nodes_df = connecting_two_dots_two_hops(sorted_dic1, sorted_dic2)

    # bind all results in to a dictionary
    result = {"connection_nodes_df":connection_nodes_df,
              "dic_ranking1":dic_ranking1,
              "dic_ranking2":dic_ranking2,
              "Temp_result_df1":Temp_result_df1,
              "Temp_result_df2":Temp_result_df2,
              "result_dic_node1":result_dic_node1,
              "result_dic_node2":result_dic_node2,
              "result_concept_node1":result_concept_node1,
              "result_concept_node2":result_concept_node2}

    #return(connection_nodes_df, dic_ranking1, dic_ranking2, Temp_result_df1, Temp_result_df2,result_dic_node1, result_dic_node2, result_concept_node1, result_concept_node2)
    return(result)


def select_result_to_analysis(sele_genes,Temp_result_df1, Temp_result_df2 ):
    
    print("selected_path: "+ ';'.join(sele_genes))
    for_plot = pd.concat([  Temp_result_df1.loc[Temp_result_df1['Object'].isin(sele_genes)],
                            Temp_result_df2.loc[Temp_result_df2['Object'].isin(sele_genes)]], axis=0)

    return(for_plot)



def plot_graph_by_predicates(for_plot):
    graph = nx.from_pandas_edgelist(for_plot, 
                                source='Subject',
                                target='Object', 
                                edge_attr=["Predicate"], 
                                create_using=nx.MultiDiGraph)


    graph_style = [{'selector': 'node[id]',
                             'style': {
                                  'font-family': 'helvetica',
                                  'font-size': '14px',
                                 'text-valign': 'center',
                                 'label': 'data(id)',
                        }},
                        {'selector': 'node',
                         'style': {
                             'background-color': 'lightblue',
                             'shape': 'round-rectangle',
                             'width': '5em',
                         }},
                        {'selector': 'edge[Predicate]',
                         'style': {
                             'label': 'data(Predicate)',
                             'font-size': '12px',
                         }},
                        {"selector": "edge.directed",
                         "style": {
                            "curve-style": "bezier",
                            "target-arrow-shape": "triangle",
                        }},
                       {"selector": "edge",
                         "style": {
                            "curve-style": "bezier",
                        }},

                    ]

    undirected = ipycytoscape.CytoscapeWidget()
    undirected.graph.add_graph_from_networkx(graph)
    undirected.set_layout(title='Path', nodeSpacing=80, edgeLengthVal=50, )
    undirected.set_style(graph_style)

    display(undirected)
    return()


def plot_graph_by_infores(for_plot):
        
    graph = nx.from_pandas_edgelist(for_plot, 
                                    source='Subject',
                                    target='Object', 
                                    edge_attr=["Infores"], 
                                    create_using=nx.MultiDiGraph)


    graph_style = [{'selector': 'node[id]',
                                'style': {
                                    'font-family': 'helvetica',
                                    'font-size': '14px',
                                    'text-valign': 'center',
                                    'label': 'data(id)',
                            }},
                            {'selector': 'node',
                            'style': {
                                'background-color': 'lightblue',
                                'shape': 'round-rectangle',
                                'width': '5em',
                            }},
                            {'selector': 'edge[Infores]',
                            'style': {
                                'label': 'data(Infores)',
                                'font-size': '12px',
                            }},
                            {"selector": "edge.directed",
                            "style": {
                                "curve-style": "bezier",
                                "target-arrow-shape": "triangle",
                            }},
                        {"selector": "edge",
                            "style": {
                                "curve-style": "bezier",
                            }},

                        ]

    undirected = ipycytoscape.CytoscapeWidget()
    undirected.graph.add_graph_from_networkx(graph)
    undirected.set_layout(title='Path', nodeSpacing=80, edgeLengthVal=50, )
    undirected.set_style(graph_style)

    display(undirected)
    return(0)


def plot_graph_by_API(for_plot):
        
    graph = nx.from_pandas_edgelist(for_plot, 
                                    source='Subject',
                                    target='Object', 
                                    edge_attr=["API"], 
                                    create_using=nx.MultiDiGraph)


    graph_style = [{'selector': 'node[id]',
                                'style': {
                                    'font-family': 'helvetica',
                                    'font-size': '14px',
                                    'text-valign': 'center',
                                    'label': 'data(id)',
                            }},
                            {'selector': 'node',
                            'style': {
                                'background-color': 'lightblue',
                                'shape': 'round-rectangle',
                                'width': '5em',
                            }},
                            {'selector': 'edge[API]',
                            'style': {
                                'label': 'data(API)',
                                'font-size': '12px',
                            }},
                            {"selector": "edge.directed",
                            "style": {
                                "curve-style": "bezier",
                                "target-arrow-shape": "triangle",
                            }},
                        {"selector": "edge",
                            "style": {
                                "curve-style": "bezier",
                            }},

                        ]

    undirected = ipycytoscape.CytoscapeWidget()
    undirected.graph.add_graph_from_networkx(graph)
    undirected.set_layout(title='Path', nodeSpacing=80, edgeLengthVal=50, )
    undirected.set_style(graph_style)

    display(undirected)
    return(0)