# Install the required libraries
#!pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

# Import the necessary libraries
import os
import time
import requests
import json
import google.auth
from google.oauth2.credentials import Credentials
from google.cloud import language_v1
import pandas as pd
from google.cloud import bigquery
import requests
from lxml import html
import chardet
from ftfy import fix_text
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
import spacy
from urllib.parse import urlparse
import openai



site_config={'cheezburger.com':'class="mu-container mu-content"','cracked.com':'class="page-content"','knowyourmeme.com':'id="main-content"'}
table_config={'categories':'sites_categories_1','entities':'sites_entities_1'}
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/yanivbenatia/nlp_test/nlp.json'
OPEN_AI_KEY='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'
class Fetcher:
    def __init__(self, site, top, fetch_type,offset=0):
        self.site = site
        self.top = top
        self.offset=offset
        self.fetch_type = fetch_type
        self.dataset_id = 'dbt_cdoyle'
        self.table_id=table_config[fetch_type]
        self.categories = []
        self.entities=[]
        self.categories_url_list = []
        self.categories_clicks_list=[]
        self.entities_url_list = []
        self.entities_clicks_list=[]
        self.entities_salience=[]
        self.entities_type=[]
        self.categories_sites=[]
        self.entities_sites=[]
        

        #init spacy 
        self.nlp = spacy.load("en_core_web_sm")

        # Set the credentials to use for the API calls
        self.credentials, self.project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

        # Set the client for the Natural Language API
        self.client = language_v1.LanguageServiceClient(credentials=self.credentials)

        # Set the client for the BigQuery API
        self.bq_client = bigquery.Client(project=self.project_id, credentials=self.credentials)
    def get_entities_from_db(self):
        query = f"""
        SELECT distinct(entity)
        FROM {self.project_id}.{self.dataset_id}.{table_config['entities']}
        WHERE site = '{self.site}' and salience>0 
        """
        query_job = self.bq_client.query(query)
        # run the query
        query_results = query_job.to_dataframe()

        # convert the results to a list
        entities = query_results.values.tolist()
        
        return entities
    def extract_entities_from_url_burt(self,url):
                # Load the pre-trained BERT model and tokenizer
        model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_map))
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        # Prepare the input
        sentence = "'12 new rick and morty memes now that the season 6 release date has been announced  2"
        input_ids = tokenizer.encode(sentence, return_tensors="pt")

        # Pass the input through the model
        output = model(input_ids)[0]

        # Use a linear layer or CRF layer on top of the output to obtain the predicted entity labels for each token
        predictions = torch.argmax(output, dim=2)
    def url_to_words(self,url):
        parsed_url = urlparse(url)

        # Extract the hostname
        title = parsed_url.path.split('/')[3].replace('-',' ').replace('-',' ').replace('/',' ')
        return title

    def extract_entities_from_url(self,url):
        
        title=self.get_page_content(url)  #url_to_words(url)
        # Process the sentence with spaCy
        doc = self.nlp(title)

        # Extract entities
        entities = [ent.text for ent in doc.ents]
        print(entities)
        return
  
    def get_embedding(self,text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    
    def get_closest_text(self,embedding, model="text-similarity-davinci-001"):
        closest_text = openai.Similarity.create(
            embedding=embedding,
            model=model
        )
        return closest_text
    def extract_entities_openai(self,url):
        
        openai.api_key=OPEN_AI_KEY
        model="text-embedding-ada-002"

        texts=self.get_page_content(url) #url_to_words(url)
        
            #dft = pd.DataFrame([texts], columns=['text'])
            #dft['ada_embedding'] = dft.text.apply(lambda x: self.get_embedding(x))
            #matrix = np.vstack(dft.ada_embedding.values)
            #n_clusters = 4
    
            #kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)
            #kmeans.fit(matrix)
            #dft['Cluster'] = kmeans.labels_
            #closest_text = self.get_closest_text(dft['ada_embedding'][0])
            #print(closest_text)

            #dft.to_csv('embedded.csv', index=False)
        return entities

        # article = ""

        # extracted_entities = extract_entities(f"Extract entities from the following text: {article}")
        # print(extracted_entities)
        # return
    
   

    # # Compute the word embeddings for the entities using a pre-trained model
    # entities_vectors = compute_word_embeddings(extracted_entities)

    # # Apply the K-Means algorithm to cluster the entities into groups
    # kmeans = KMeans(n_clusters=5, random_state=0).fit(entities_vectors)

    # # Print the cluster labels for each entity
    # print(kmeans.labels_)
    # Please note that the above example is a very basic implementation, and you should implement it based on the complexity of your data and the resources available. Additionally, the quality of the result will also depend on the quality and amount of data you provide.





    def fetch(self):
        # Query the top 20 URLs from the source table, ordered by clicks
        query = f"""
        SELECT url, sum(clicks) as clicks
        FROM `literally-analytics.rivery.search_console_discover_by_url`
        WHERE property_id = 'sc-domain:{self.site}'
        and date > '2023-01-01'
        group by 1
        ORDER BY 2 DESC
        LIMIT {self.top} offset {self.offset}
        """
        query_job = self.bq_client.query(query)
        urls = query_job.to_dataframe()

        # Iterate through the URLs and extract the categories or entities
        for index, row in urls.iterrows():
            url = row['url']
            clicks = row['clicks']
            page_content=self.get_page_content(url)
            if not page_content:
                 print("skip url:",url)
                 continue
            document = language_v1.Document(
                content=page_content ,
                type=language_v1.Document.Type.HTML
                )
            
            if self.fetch_type == 'categories':
                try:
                    response = self.client.classify_text(request={"document":document})
                except Exception:
                    print('issue analyzing url:',url)
                    continue
                self.categories.extend([category.name for category in response.categories])
                self.categories_url_list.extend([url] * len(response.categories))
                self.categories_clicks_list.extend([clicks] * len(response.categories))
                self.categories_sites.extend([self.site] * len(response.categories))
            elif self.fetch_type == 'entities':
                try:
                    response = self.client.analyze_entities(request={"document":document})
                except Exception:
                    print('issue analyzing url:',url)
                    continue
                self.entities.extend([entity.name for entity in response.entities])
                self.entities_type.extend([entity.type.name for entity in response.entities])
                self.entities_salience.extend([entity.salience for entity in response.entities])
                self.entities_url_list.extend([url] * len(response.entities))
                self.entities_clicks_list.extend([clicks] * len(response.entities))
                self.entities_sites.extend([self.site] * len(response.entities))
          
            time.sleep(0.5)

    def get_results(self):
        if self.fetch_type == 'categories':
            return {
                'category': self.categories,
                'url': self.categories_url_list,
                'clicks': self.categories_clicks_list,
                'site': self.categories_sites
            }
        elif self.fetch_type == 'entities':
            return {
                'entity': self.entities,
                'url': self.entities_url_list,
                'type':self.entities_type,
                'salience':self.entities_salience,
                'clicks': self.entities_clicks_list,
                'site': self.entities_sites
            }

    def get_page_content(self,url):
        
        try:
            #set header to mobile to get mobile layout
            headers = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1'}
            
            # Send a request to the URL
            r = requests.get(url,headers=headers)

            # Detect the encoding of the response content
            # result = chardet.detect(r.content)
            # encoding = result['encoding']

            # Get the text content from the response
            text = r.content.decode('utf-8')

            # Fix the text encoding using the ftfy library
            fixed_text = fix_text(text)

            # # If the encoding is not UTF-8, convert it to UTF-8
            # if encoding.upper() != 'UTF-8':
            #     r.encoding = encoding
            #     content = r.content.decode(encoding)
            #     r.content = content.encode('utf-8')

            # Parse the HTML content
            tree = html.fromstring(fixed_text)

            # Find the element with the class "page-content"
            path=site_config[self.site]
            #'//h1[1]/text()'
            #page_content = tree.xpath('//h1[1]/text()')[0]
            #page_content = tree.xpath('//*[@{}]'.format(path))[0]
            page_content=tree.xpath('//h2[@id="about"]/following-sibling::p[1]')[0]
            # return ' '.join(page_content)
            # Return the content inside the element
            return page_content.text_content()
        except Exception:
            return False
        
    #load result into a BQ table 
    def insert_result_to_db(self,results):
        
        print(f"Loading {self.site} {self.fetch_type} into {self.project_id}.{self.dataset_id}.{self.table_id}")
        
        # Convert the results to a dataframe
        df = pd.DataFrame.from_dict(results)

        # Set the client for the BigQuery API
        self.bq_client  = bigquery.Client(project=self.project_id, credentials=self.credentials)
        # Get the table reference
         # Get the table reference
        table_ref =  self.bq_client .dataset(self.dataset_id).table(self.table_id)

        # Try to get the table
        try:
            table = self.bq_client.get_table(table_ref)
        except:
            # If the table doesn't exist, set it to False
            table = False

        if table:
            # If the table exists, append the data to it
            self.bq_client .load_table_from_dataframe(df, table_ref).result()
        else:
            # If the table doesn't exist, create it and insert the data
            table = bigquery.Table(table_ref)
            self.bq_client.create_table(table)
            self.bq_client.load_table_from_dataframe(df, table_ref).result()
    
    
    def get_bert_vectors(self,entities):
        # Initialize the BERT model and tokenizer
        
        model = BertModel.from_pretrained('bert-base-uncased',cache_dir='~/Library/Caches/transformers/')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='~/Library/Caches/transformers/')

        # Encode the entities using BERT
        input_ids = [tokenizer.encode(e, add_special_tokens=True) for e in entities]
        input_ids = torch.tensor(input_ids)

        # Get the pooled output of the BERT model
        with torch.no_grad():
            _, pooled_output = model(input_ids)

        return pooled_output

    def cluster_entities(self,entities):
        # Train a Word2Vec model on the entities
        model = Word2Vec(entities, vector_size=100, window=5, min_count=1)

        # Extract the word embeddings for the entities
        X = model.wv.vectors

        # Perform DBSCAN clustering on the word embeddings
        db = DBSCAN(eps=0.8, min_samples=10)
        clusters = db.fit_predict(X)

        # Group the entities by cluster
        grouped_entities = {}
        for i, cluster in enumerate(clusters):
            if cluster not in grouped_entities:
                grouped_entities[cluster] = []
            grouped_entities[cluster].append(entities[i])

        return grouped_entities
    def cluster_entities_bert(self, entities):
        bert_vectors = self.get_bert_vectors(entities)
        bert_vectors = bert_vectors.detach().numpy()
        # Perform DBSCAN clustering on the BERT vectors
        db = DBSCAN(eps=0.3, min_samples=2)
        clusters = db.fit_predict(bert_vectors)

        # Group the entities by cluster
        grouped_entities = {}
        for i, cluster in enumerate(clusters):
            if cluster not in grouped_entities:
                grouped_entities[cluster] = []
            grouped_entities[cluster].append(entities[i])
        return grouped_entities

def run(site, top, fetch_type,offset=0):
    fetcher = Fetcher(site, top, fetch_type,offset)
    fetcher.fetch()
    results = fetcher.get_results()
    df = pd.DataFrame(fetcher.entities, columns=["Entity"])
    df = pd.get_dummies(df, columns=["Entity"])

    # Use k-means to cluster the entities
    kmeans = KMeans(n_clusters=3, random_state=0).fit(df)
    df["Topic"] = kmeans.labels_
    # Find the top N entities within each topic
    N = 10
    top_entities = df.groupby("Topic").sum().apply(lambda x: x.sort_values(ascending=False).head(N), axis=1)

    # Generate labels for each topic
    topic_labels = top_entities.apply(lambda x: ', '.join(x.index), axis=1)
    df["Topic Label"] = df["Topic"].map(topic_labels)

    # # Generate labels for each topic
    # topic_counts = df["Topic"].value_counts()
    # topic_labels = {topic: f"Topic {topic+1}" for topic in topic_counts.index}
    # df["Topic Label"] = df["Topic"].map(topic_labels)


    print(df)
    fetcher.insert_result_to_db(results)
    return




def main(top,offset=0):
    for i in table_config:
        for x in site_config.keys():
            print("Analyzing {} {} on {} urls with offset {}".format(x,i,top,offset))
            run(x,top,i,offset)
    print("done!")

#main(100,0)
#run('knowyourmeme.com',10,'categories')
run('knowyourmeme.com',10,'entities')



#fetcher = Fetcher('cheezburger.com', 50, 'entities',0)
#fetcher.extract_entities_openai('https://knowyourmeme.com/memes/jrpg-taco-bell')
#entities=fetcher.get_entities_from_db()
#entities_clusters=fetcher.cluster_entities_bert(entities)
#print(entities_clusters)


