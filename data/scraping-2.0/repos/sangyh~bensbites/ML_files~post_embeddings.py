import pandas as pd
from utils import utils
import json
import os
import openai
import pinecone

ROOT_DIR = os.path.abspath('/home/sangy/Desktop/SideProjects/BensBites/bensbites')
path = ROOT_DIR + '/ML_files'

class extract_posts():
    def __init__(self):
        pass    
    
    def scrape(self, slugs):
        ### read posts json
        f = open(os.path.join(path,'bensbites_scraped.json'))
        data = json.load(f)
        f.close()  

        ### compile json to df
        cols=['url','section','item']
        df = pd.DataFrame(columns=cols)
        for slug in slugs:
            post_url = 'https://www.bensbites.co/p/' + slug
            tmp = utils.compress_json(post_url, data)
            df = pd.concat([df,tmp],ignore_index=True,axis=0) 

        df['item_text'] = df['item'].apply(lambda x:x[0])
        df['item_text_len'] = df['item_text'].apply(len)
        df = df[df.item_text_len>0]
        df['item_url'] = df['item'].apply(lambda x:x[1])


        #remove these generic texts
        tmp = pd.DataFrame(df.item_text.value_counts()).rename(columns={'item_text':'textfreq'})
        df = df.join(tmp, on='item_text', how='left')
        df=df[df.textfreq==1]
        
        # df['ada_embedding'] = df.item_text.apply(lambda x: utils.get_embedding(x, model='text-embedding-ada-002'))
        # df['cluster_id']= df.ada_embedding.apply(lambda x: utils.predict_cluster(x))

        # create post embeddings in pinecone db
        utils.create_post_embeddings(df)        
            
        f = open(os.path.join(path,'topicid2labels.json'))
        topicid2labels = json.load(f)
        f.close()  

        #df[['topic','total_tokens']]  = df.item_text.apply(utils.assign_topics_gpt3, args=(topicid2labels,))
        
        ### read existing embeddings csv file
        if os.path.isfile(os.path.join(path,'embedded_bensbites.csv')):
            curr_df = pd.read_csv(os.path.join(path,'embedded_bensbites.csv'))
            curr_df = pd.concat([curr_df, df], axis=0)
        else:
            print('Creating new embeddings file')
            curr_df = df

        curr_df.to_csv(os.path.join(path,'embedded_bensbites.csv'), index=False)
