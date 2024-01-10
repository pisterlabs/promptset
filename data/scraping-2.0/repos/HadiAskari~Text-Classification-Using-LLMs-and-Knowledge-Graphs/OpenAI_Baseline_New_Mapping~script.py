import argparse
import os
import requests
import json
import openai
import pickle as pkl
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from time import sleep
from tqdm.auto import tqdm
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import plotly.express as px


def create_embeddings(azureconfig,item,total_embeddings):    
    deployment_name= azureconfig['deployment_name']

    response = openai.Embedding.create(
        input=item,
        engine=deployment_name
    )
    embeddings = response['data'][0]['embedding']
    total_embeddings.append(embeddings)


def tSNE_similarity(maf_embeddings,iab_embeddings):
    top_n_labels_new=[]
    top_n_cosines_new=[]
    top_n_nodes=[]
    top_n_labels=[]
    top_n_cosines=[]
    tsne_embeddings=[]



    for i in tqdm(range(len(maf_embeddings))):
        similarities=[]
        for j in range(len(iab_embeddings)):
            cos_sim=cosine_similarity(np.array(iab_embeddings[j]).reshape(1,-1),np.array(maf_embeddings[i]).reshape(1,-1))
            similarities.append([call[160+i],call[j],cos_sim])
        
        similarities.sort(key=lambda x: x[2], reverse=True)

        new_similarities=[x for x in similarities if x[2]>=0.765]   #LOW THRESHOLD TO SEE GENERAL CLUSTER VALUES

        if len(new_similarities)==0:
            new_similarities.append(['No_Pred','No_Pred',[[0]]])
        
        temp_iab_labels=[]
        temp_cosines=[]


        for items in new_similarities:
            tsne_embeddings.append(maf_embeddings[i])

            temp_iab_labels.append(items[1])
            temp_cosines.append(items[2][0])        
            top_n_labels_new.append(items[1])
            top_n_cosines_new.append(items[2][0])
            top_n_nodes.append(nodes[i])



        
        top_n_labels.append(temp_iab_labels)
        top_n_cosines.append(temp_cosines)
    
    return tsne_embeddings, top_n_labels_new, top_n_nodes

def calculate_tSNE_clusters(iab_labels, df_labels_embeddings):
    results={}
    for label in tqdm(iab_labels):
        mask = df_labels_embeddings['labels'].isin([label])
        df_tier1 = df_labels_embeddings[mask]
        if df_tier1.shape[0]//2<50:         #setting custom perplexity numbers
            perp=df_tier1.shape[0]//2       
        else:
            perp=50
        to_TSNE_list=df_tier1['embeddings'].to_list()
        to_TSNE_arr=np.array(to_TSNE_list)
        tsne=TSNE(n_components=2, random_state=42, perplexity=perp)
        X_embedded = tsne.fit_transform(to_TSNE_arr)
        results[label]=tsne.kl_divergence_
    
    return results

def get_problematic(kl_thresh, results):
    problematic=[]
    #new_problematic=[]
    for k,v in results.items():
        if  v>kl_thresh:   #Here v is an heuristic that I set after a bit of trial and error
            print(k,v)
            problematic.append(k)
    
    return problematic

def openAI_similarity(maf_embeddings,iab_embeddings,TopN,upper,lower,parent_bool):
    top_n_labels_new=[]
    top_n_cosines_new=[]
    top_n_nodes=[]
    top_n_labels=[]
    top_n_cosines=[]

    tsne_embeddings=[]



    for i in tqdm(range(len(maf_embeddings))):
        similarities=[]
        for j in range(len(iab_embeddings)):
            cos_sim=cosine_similarity(np.array(iab_embeddings[j]).reshape(1,-1),np.array(maf_embeddings[i]).reshape(1,-1))
            similarities.append([call[160+i],call[j],cos_sim])
        
        similarities.sort(key=lambda x: x[2], reverse=True)

        
        #low initial threshold
        similarities=[x for x in similarities if x[2]>=0.765]
        new_similarities=[]

        # new_similarities=[x for x in similarities if x[2]>=0.8065]

        # #higher threshold for problematic keywords, high threshold for general keywords for accuracy
        for x in similarities:
            if len(new_similarities)>=TopN and x[2]<upper: #TOP N Thresholding
                break
            if x[2]>=upper and x[1] in problematic:  # Problematic Keyword Threshold
                new_similarities.append(x)
            elif x[2]>=lower and x[1] not in problematic:  # Non Problematic Keyword Threshold
                new_similarities.append(x)
            else:
                continue
        
        #adding atleast 1 adword
        if len(new_similarities)==0:
            while len(new_similarities)<1:
                new_similarities.append(similarities[len(new_similarities)])
 

        #force adding all parent adwords if child is predicted
        if parent_bool==True:
            new_sim_adwords=[]
            for items in new_similarities:
                new_sim_adwords.append(items[1])

            count=0
            for items in new_similarities:
                if count==5:
                    break
                try:
                    parent_list=child_parent_dic[items[1]]
                    for parent in parent_list:
                        if parent not in new_sim_adwords:
                            new_sim_adwords.append(parent)
                            j=call.index(parent)
                            cos_sim=cosine_similarity(np.array(iab_embeddings[j]).reshape(1,-1),np.array(maf_embeddings[i]).reshape(1,-1))
                            if cos_sim>0.765: #Only adding parents if cosine similarity crosses a certain threshold
                                new_similarities.append([call[160+i],call[j],cos_sim])
                except Exception as e:
                    #print(e)
                    pass

                count+=1
        
            
        temp_iab_labels=[]
        temp_cosines=[]

        count=0
        for items in new_similarities:
            if count>=TopN and items[2][0][0]<upper:
                break
            tsne_embeddings.append(maf_embeddings[i])
            temp_iab_labels.append(items[1])
            temp_cosines.append(items[2][0])        
            top_n_labels_new.append(items[1])
            top_n_cosines_new.append(items[2][0][0])
            top_n_nodes.append(nodes[i])
            count+=1

        
        top_n_labels.append(temp_iab_labels)
        top_n_cosines.append(temp_cosines)

    return top_n_nodes,top_n_labels_new,top_n_cosines_new


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--i", '--internal', type=str, help="path to .txt list of internal keywords (MAF or RAM)", required=True)
    parser.add_argument("--e", '--external', type=str, help="path to .txt list of external keywords (IAB or GT)", required=True)
    parser.add_argument("--u", "--upper_thresh",type=float, help="Upper Threshold", required=True)
    parser.add_argument("--l", "--lower_thresh",type=float, help="Lower Threshold", required=True)
    parser.add_argument("--n", "--TopN_Thresh",type=float, help="Top N Threshold", required=True)
    parser.add_argument("--p", "--parent_bool", help="if parent child dict provided or not BOOL", type=bool, required=True)
    parser.add_argument('--k', "--KL_Divergence_Threshold",type=float, help="KL Threshold", required=True)

    #Checking if keys are working properly

    with open('azure-configuration.json') as inputfile:
        azureconfig = json.load(inputfile)
    openai.api_key = azureconfig['key'] 
    openai.api_base = azureconfig['endpoint'] 
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15' # this may change in the future

    deployment_name= azureconfig['deployment_name']

    response = openai.Embedding.create(
        input="Your text string goes here",
        engine=deployment_name
    )
    embeddings = response['data'][0]['embedding']

    print('Check to see if OpenAI Keys loaded properly: {}'.format(len(embeddings)))

    if not embeddings:
        raise "OpenAI keys not valid"
    
    args=parser.parse_args()

    with open(args.i, 'r') as f: #Loading MAF Labels
        check=f.readlines()
    
    check=[x.split('\n')[0] for x in check] 

    maf=check

    #print(len(maf))

    with open(args.e, 'r') as f: #Loading IAB Labels
        check1=f.readlines()

    check1=[x.split('\n')[0] for x in check1] 

    iab_labels=check1

    #print(len(iab_labels))

    call=iab_labels+maf
  

    total_embeddings=[]

    print('Creating Embeddings')

    for items in tqdm(call):
        try:
            items=items.replace("&", 'and')
        except:
            pass

        create_embeddings(azureconfig,items,total_embeddings)
    
    print('length of total embeddings: {}'.format(len(total_embeddings)))

    iab_embeddings=total_embeddings[0:160]  #Change this if you want to test a new external keyword set
    maf_embeddings=total_embeddings[160:]

    nodes=maf

    print('Calculating initial similarity Embeddings')
    tsne_embeddings, top_n_labels_new, top_n_nodes = tSNE_similarity(maf_embeddings,iab_embeddings)

    dic={'embeddings':tsne_embeddings, 'labels':top_n_labels_new, 'item': top_n_nodes}
    df_labels_embeddings=pd.DataFrame.from_dict(dic)

    print('Calculating tSNE Clusters')
    results=calculate_tSNE_clusters(iab_labels, df_labels_embeddings)

    problematic=get_problematic(args.k, results)

    #TSNE PART OVER

    if args.p == True:
        with open("data/child_parent_dic.pkl", 'rb') as f:
            child_parent_dic=pkl.load(f)

    print('Calculating final mapping')
    top_n_nodes,top_n_labels_new,top_n_cosines_new=openAI_similarity(maf_embeddings,iab_embeddings,args.n,args.u,args.l,args.p)

    dic={'item':top_n_nodes, 
'predictions': top_n_labels_new, 'cosine_predicted': top_n_cosines_new}
    
    df_MAF_new=pd.DataFrame.from_dict(dic)

    df_MAF_new.to_csv('CHECK_MAPPING.csv')






    

    
    

