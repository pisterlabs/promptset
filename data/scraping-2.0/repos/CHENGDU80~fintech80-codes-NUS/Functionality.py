#!/usr/bin/env python
# coding: utf-8

# ### Libraries 

# In[546]:


import os
import pandas as pd

import pandas as pd
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import textwrap
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.cluster import KMeans
import sklearn.cluster
import matplotlib.pyplot as plt
from InstructorEmbedding import INSTRUCTOR
import sklearn.cluster
from PyPDF2 import PdfReader
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import re
from sentence_transformers import SentenceTransformer
import faiss
import time
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import joblibfrom transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from selenium import webdriver
import pandas as pd
from bs4 import BeautifulSoup
import time
import plotly.express as px
import yfinance as yf
from datetime import datetime
import requests


# In[ ]:


os.environ["OPENAI_API_KEY"]='key_hide'
llm = OpenAI(temperature=0)


# In[ ]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_multi = INSTRUCTOR('hkunlp/instructor-base')  
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


# In[532]:


df=pd.read_csv("AAPL_news_yahoo_sent.csv")


# #### web scrapping

# In[ ]:


def webscrapping(company,url):
    driver = webdriver.Chrome()
    driver.get(url)
    for i in range(20):
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, 99999);")
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html")   
    df = pd.DataFrame(columns=['title','desc','url'])
    title = []
    url = []
    for i in soup.findAll("h3", class_="Mb(5px)"):
        title.append(i.text)
        url.append("https://finance.yahoo.com"+i.findChild('a').get('href'))
    desc = []
    for j in soup.findAll("p", {"class":["Fz(14px) Lh(19px) Fz(13px)--sm1024 Lh(17px)--sm1024 LineClamp(3,57px) LineClamp(3,51px)--sm1024 M(0)", "Fz(14px) Lh(19px) Fz(13px)--sm1024 Lh(17px)--sm1024 LineClamp(2,38px) LineClamp(2,34px)--sm1024 M(0)", "Fz(14px) Lh(19px) Fz(13px)--sm1024 Lh(17px)--sm1024 LineClamp(2,38px) LineClamp(2,34px)--sm1024 M(0) D(n)--sm1024 Bxz(bb) Pb(2px)"]}):
        desc.append(j.text)
    date = []
    for k in soup.findAll('div', class_='C(#959595) Fz(11px) D(ib) Mb(6px)'):
        date.append(k.text)

    df['title'] = title
    df['desc'] = desc
    df['url'] = url
    df['source'] = date
    
    return df 


# In[ ]:


today = pd.to_datetime(datetime.date.today())
def create_date(row):
    temp = row['source']
    if 'minute' in temp or 'hour' in temp:
        return today
    elif 'yesterday' in temp:
        return today - pd.to_timedelta(1,'days')
    elif '2' in temp:
        return today - pd.to_timedelta(2,'days')
    elif '3' in temp:
        return today - pd.to_timedelta(3,'days')    
    elif '4' in temp:
        return today - pd.to_timedelta(4,'days')    
    elif '5' in temp:
        return today - pd.to_timedelta(5,'days')
    elif '6' in temp:
        return today - pd.to_timedelta(6,'days')
    elif '7' in temp:
        return today - pd.to_timedelta(7,'days')    
    elif '8' in temp:
        return today - pd.to_timedelta(8,'days')    
    elif '9' in temp:
        return today - pd.to_timedelta(9,'days')
    elif '10' in temp:
        return today - pd.to_timedelta(10,'days')
    elif '11' in temp:
        return today - pd.to_timedelta(11,'days')    
    elif '12' in temp:
        return today - pd.to_timedelta(12,'days')    
    elif '13' in temp:
        return today - pd.to_timedelta(13,'days')
df['date'] = df.apply(create_date,axis=1)
df['source'] = df['source'].str.split('•').str[0]


# ### Finbert

# In[ ]:


def finbert(df):
    df['feed'] = df['title'] + ". " + df['desc']
    sent_val = list()
    conf_score = list()
    pipe = pipeline("text-classification", model="ProsusAI/finbert")
    for x in df['feed']:
        p = pipe(x)[0]
        val = p['label']
        conf = p['score']
        print(x, '----', val)
        print('#######################################################')
        sent_val.append(val)
        conf_score.append(conf)
    df['sentiment'] = sent_val
    df['confidence'] = conf_score
    df = df.drop(columns=['feed'])   
    
    return df 


# In[ ]:


sentiment_analysis = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

news_text = "Markets will gyrate right up until the Fed announces its interest-rate decision. But don't forget: Apple reports quarterly results on Thursday."
sentiment_result = sentiment_analysis(news_text)

print(sentiment_result)


# ## impact analysis for negative and positive

# In[217]:


def neagtive_impact_analysis(df):
    df[df["date"]=='2023-10-29']
    negative_news=df[df['sentiment']=='negative']
    negative_news = negative_news.sort_values(by='confidence', ascending=False)
    negative_news=negative_news.head(1)
    negative_news_list=negative_news['desc'].tolist()
    negative_impact_list=[]
    negative_impact_dict={}                 
    for i in negative_news_list:
        
        #print(str(i))
        #demo_template='''Please clasify this content {docs} into three categories: financial-related contents, operational-related contents, brand reputational-related contents, and others.'''
        #demo_template='''Given that {impact}, what impact can this lead on NVIDIA Corporation (NVDA)? Please give me 3 most important bullet points and each one with clear and concise reasoning.'''
        #demo_template='''Given that {impact}, what impact can this lead on Apple Inc(AAPL)? Please give me 3 most important bullet points and each one with clear and concise reasoning.'''
        demo_template='''Given that {impact}, what impact can this lead on Microsoft Corporation (MSFT) stock return ? Please give me 3 most important bullet points and each one with clear and concise reasoning.'''
        prompt=PromptTemplate(
            input_variables=['impact'],
            template=demo_template
            )

        prompt.format(impact=i)
        chain1=LLMChain(llm=llm,prompt=prompt)
        news_impact_output=chain1.run(i)
        print(news_impact_output)
        negative_impact_list.append(news_impact_output)
        negative_impact_dict[i]=news_impact_output
        
    data_negative = {'negative_news': negative_news_list, 'negative_impact': negative_impact_list}
    df_negative = pd.DataFrame(data_negative)

        # Optionally, set column names
    df_negative.columns = ['negative_news', 'negative_impact']
    df_negative['negative_impact'] = df_negative['negative_impact'].str.replace('\n', '', regex=True)
     
    return df_negative ,negative_impact_dict              
                     
                     
                     


# In[553]:


microsoft_dataframe,microsoft_negative_dict=neagtive_impact_analysis(df)


# In[212]:


df=pd.read_csv("NVDA_news_yahoo_sent.csv")


# In[ ]:





# In[211]:


def positive_impact_analysis(df):
    positive_news=df[df['sentiment']=='positive']
    positive_news = positive_news.sort_values(by='confidence', ascending=False)
    positive_news=positive_news.head(3)
    positive_news_list=positive_news['desc'].tolist()                   
    positive_impact_list=[] 
    positive_impact_dict={}   
    for i in positive_news_list:
        #print(str(i))
        #demo_template='''Please clasify this content {docs} into three categories: financial-related contents, operational-related contents, brand reputational-related contents, and others.'''
        demo_template='''Given that {impact}, what impact can this lead on NVIDIA Corporation (NVDA)? Please give me 3 most important bullet points and each one with clear and concise reasoning.'''
        #demo_template='''Given that {impact}, what impact can this lead on Apple Inc(AAPL)? Please give me 3 most important bullet points and each one with clear and concise reasoning.'''
        #demo_template='''Given that {impact}, what impact can this lead on Microsoft Corporation (MSFT) ? Please give me 3 most important bullet points and each one with clear and concise reasoning.''' 
        
        #demo_template='''Given that {impact}, what impact can this lead to? Please give me 3 most important bullet points and each one with clear and concise reasoning.'''
        prompt=PromptTemplate(
            input_variables=['impact'],
            template=demo_template
            )

        prompt.format(impact=i)
        chain1=LLMChain(llm=llm,prompt=prompt)
        news_impact_output=chain1.run(i)
        print(news_impact_output)
        positive_impact_list.append(news_impact_output)
        positive_impact_dict[i]=positive_impact_dict
        
    data_positive = {'positive_news': positive_news_list, 'positive_impact': positive_impact_list}
    df_positive = pd.DataFrame(data_positive)

        # Optionally, set column names
    df_positive.columns = ['positive_news', 'positive_impact']
    df_positive['positive_impact'] = df_positive['positive_impact'].str.replace('\n', '', regex=True)
         
    return df_positive,positive_impact_dict               
                     
                     


# In[552]:


nvidia_pos_dataframe,nvidia_positive_dict=positive_impact_analysis(df)


# In[512]:


df=pd.read_csv("miltview_MSFT.csv")


# ### mutli_angle

# In[535]:


def multi_angle_summary(df,n,company_name):
    model_multi = INSTRUCTOR('hkunlp/instructor-base')
    embeddings = model_multi.encode(df['desc'])
    clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=5)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_
    print(cluster_assignment)
    df['clusters']=cluster_assignment
    key={}
    df_1=df.head(50)
    df_1=df_1[df_1["clusters"]==n]
    prompt_template="""Based on the given text {text}, give me one summary of the text, be consistent and logical."""
   # prompt=PromptTemplate.from_template(prompte_template)
    
    BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template, 
                            input_variables=["text"])
    sentence_list = df_1['desc'].tolist()
    from langchain.docstore.document import Document
    docs = [Document(page_content=t) for t in sentence_list]
    docs
    chain = load_summarize_chain(llm, 
                             chain_type="stuff", 
                             prompt=BULLET_POINT_PROMPT)

    output_summary = chain.run(docs)

    wrapped_text = textwrap.fill(output_summary, 
                                 width=100,
                                 break_long_words=False,
                                 replace_whitespace=False)
    #text_without_newlines = output_summary.replace('\n', '')
    key[company_name]=output_summary
    
    return key


# In[ ]:


apple_multi_view1=(df,0,'Apple')
apple_multi_view1=(df,1,'Apple')
apple_multi_view1=(df,2,'Apple')
apple_multi_view1=(df,3,'Apple')
apple_multi_view1=(df,4,'Apple')


# In[526]:


df_multi=df[df["clusters"]==4]


# In[ ]:


data = {
    'cluster0': apple_multi_view1,
    'cluster1': apple_multi_view2,
    'cluster2': apple_multi_view3,
    'cluster3': apple_multi_view4,
    'cluster4': apple_multi_view5
}

df_mul = pd.DataFrame(data)
df_mutli= df_mul.transpose()
df_mutli


# In[ ]:





# ## summarization

# In[98]:


#chain = load_summarize_chain(llm, chain_type="stuff")


# In[478]:


df=pd.read_csv("df_macro_sent.csv")


# In[487]:


def summary(df):
    key={}
    df_1=df.head(50)
    #prompt_template = """Write a concise bullet point summary of minimum 200 and maximum 250 words of the entity NVIDIA Corporation (NVDA) of the following:
    #{text}
    #CONSCISE SUMMARY IN BULLET POINTS:"""
    #prompt_template ="""Write a concise bullet point summary maximum of 50 words of the industry finance of the following:
    #{text}
    #CONSCISE SUMMARY IN BULLET POINTS:"""
    prompt_template = """Write a summary of the macro economic enivronment minimum of 200 words based on the following news articles:
    {text}
    SUMMARY:"""
    prompt=PromptTemplate.from_template(prompt_template)
    
    BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template, 
                            input_variables=["text"])
    sentence_list = df_1['Description'].tolist()
    #sentence_list = df_1['desc'].tolist()
    from langchain.docstore.document import Document
    docs = [Document(page_content=t) for t in sentence_list]
    docs
    chain = load_summarize_chain(llm, 
                             chain_type="stuff", 
                             prompt=BULLET_POINT_PROMPT)

    output_summary = chain.run(docs)

    wrapped_text = textwrap.fill(output_summary, 
                                 width=100,
                                 break_long_words=False,
                                 replace_whitespace=False)
    #text_without_newlines = output_summary.replace('\n', '')
    key['200']=output_summary
    
    return key


# In[ ]:


macro_50=summary(df)
macro_200=summary(df)
macro_300=summary(df)
series1 = pd.Series(macro_50)
series2 = pd.Series(macro_200)
#series3 = pd.Series(nvidia_300)
df_43= pd.concat([series1, series2], axis=0)
df_43.to_csv("macro_summary_dataframe.csv")


# ## Chatbot question and answer

# In[549]:


def chatbot_embeddings():
    pdfreader = PdfReader('data.pdf')
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 250,
    chunk_overlap = 30,
    length_function = len
    )

    text_chunks = text_splitter.split_text(raw_text)
    df_texts = pd.DataFrame({'id': range(1, len(text_chunks) + 1), 'text': text_chunks})
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(df_texts['text'].tolist())
    id_array = df_texts['id'].to_numpy()
    index = faiss.IndexIDMap(faiss.IndexFlatIP(384))
    index.add_with_ids(embeddings, id_array)
    faiss.write_index(index, 'finance_index')
    return df_texts


# In[ ]:


df_texts=chatbot_embeddings()


# In[550]:


def chatbot_query(query,df_texts):
    index = faiss.read_index('finance_index')
    t=time.time()
    query_vector = model.encode([query])
    k = 20
    top_k = index.search(query_vector, k)
    ids_to_match=ans[1][0].tolist()
    matched_texts = df_texts[df_texts['id'].isin(ids_to_match)]['text'].tolist()
    matched_texts
    from langchain.docstore.document import Document
    docs = [Document(page_content=t) for t in matched_texts]
    chain = load_qa_chain(llm, chain_type="stuff")
    answer=chain.run(input_documents=docs, question=ans)
    return answer


# In[ ]:


chatbot_query("what is the gross margin for the june quarter?",df_texts)


# ### Retrace news 

# In[ ]:


def search_retrace(query, model, index, k=10, threshold=0.5):
    # Assuming the vectors are already normalized and the index is an IndexFlatIP
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)  # Normalize the query vector if it isn't already
    D, I = index.search(query_vector, k)
    
    # Apply the threshold to filter results
    mask = D[0] >= threshold
    I_filtered = I[0][mask]
    D_filtered = D[0][mask]
    
    return D_filtered, I_filtered


# In[ ]:


def retrace():
    df=pd.read_csv('AAPL_news_yahoo_sent.csv')
    df['date'] = pd.to_datetime(df['date'])
    df_curr = df[df['date'].dt.day == 29]
    df_curr = df_curr.head(4)
    df_curr
    start_date = pd.Timestamp('2023-10-20 00:00:00')
    end_date = pd.Timestamp('2023-10-29 00:00:00')
    df_hist = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df_hist['id'] = range(1, len(df_hist) + 1)
    id_array = df_hist['id'].to_numpy()
    embeddings = model.encode(df_hist['desc'].tolist())
    index = faiss.IndexIDMap(faiss.IndexFlatIP(1024))
    index.add_with_ids(embeddings, id_array)
    added_descriptions = set()
    results = [] 
    for desc in df_curr['desc']:
        D, ids_to_match = search(desc, model, index)
        if len(ids_to_match) == 0:
            continue  
    matched_texts = df_hist.loc[df_hist['id'].isin(ids_to_match), 'desc'].tolist()
    matched_texts = [text for text in matched_texts if text not in added_descriptions]
    added_descriptions.update(matched_texts)
    filtered_df = df_hist[df_hist['id'].isin(ids_to_match)]
    new_df = filtered_df[['desc', 'date','sentiment','confidence']].copy() 
    new_df = new_df[~new_df['desc'].isin([desc])]
    new_df['original_desc'] = desc
    results.append(new_df)
    final_results_df = pd.concat(results, ignore_index=True)

    date_range = pd.date_range(start=final_results_df['date'].min(), end=final_results_df['date'].max())


# In[551]:


def plot_retrace():
    date_range = pd.date_range(start=final_results_df['date'].min(), end=final_results_df['date'].max())
    sentiment_counts = final_results_df.groupby(['date', 'sentiment']).size().reset_index(name='count')
    complete_sentiment_counts = (
        sentiment_counts.set_index(['date', 'sentiment'])
        .reindex(pd.MultiIndex.from_product([date_range, final_results_df['sentiment'].unique()], names=['date', 'sentiment']), fill_value=0)
        .reset_index()
    )

    color_map = {
        'negative': 'red',
        'neutral': 'grey',
        'positive': 'green'
    }
    ordered_sentiments = ['negative', 'neutral', 'positive']
    complete_sentiment_counts['sentiment'] = pd.Categorical(complete_sentiment_counts['sentiment'], categories=ordered_sentiments, ordered=True)
    fig = px.line(
        complete_sentiment_counts,
        x='date',
        y='count',
        color='sentiment',
        title='Sentiment Counts Over Time For Apple Stocks',
        labels={'count': 'Number of Occurrences', 'date': 'Date'},
        color_discrete_map=color_map  # Use the color map for discrete colors based on sentiment
    )
    fig.update_traces(
        mode='lines+markers',
        marker=dict(size=8)
    )
    for i, sentiment in enumerate(ordered_sentiments):
        fig.for_each_trace(
            lambda t, pattern=i: t.update(line=dict(dash=['solid', 'dot', 'dash'][pattern])) if t.name == sentiment else (),
        )

    fig.update_yaxes(range=[0, complete_sentiment_counts['count'].max() + 1])
    fig.show()


# In[ ]:





# ## Value chain

# In[ ]:


df = pd.read_csv('AAPL_news_yahoo_sent.csv')
def query(payload):
    API_URL = "https://api-inference.huggingface.co/models/Gladiator/microsoft-deberta-v3-large_ner_conll2003"
    headers = {"Authorization": "Bearer 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"}
    response = requests.post(API_URL, headers=headers, json=payload)
    for i in range(107,170):
          output = query({
            "inputs": df['feed'][i],
          })
          loc_words = [entry['word'] for entry in output if entry['entity_group'] == 'ORG']
          coy_list.append(list(dict.fromkeys(loc_words)))
          print(i)
    
    return response.json()

API_URL = "https://api-inference.huggingface.co/models/Gladiator/microsoft-deberta-v3-large_ner_conll2003"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    API_URL = "https://api-inference.huggingface.co/models/Gladiator/microsoft-deberta-v3-large_ner_conll2003"
    headers = {"Authorization": "Bearer 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"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# coy_list = []
for i in range(107,170):
  output = query({
    "inputs": df['feed'][i],
  })
  loc_words = [entry['word'] for entry in output if entry['entity_group'] == 'ORG']
  coy_list.append(list(dict.fromkeys(loc_words)))
  print(i)
    
filtered_list = [sublist for sublist in coy_list if 'Bloomberg' not in sublist]

# Step 1: Flatten the list of lists into a single list
flat_list = [company for sublist in filtered_list for company in sublist]

# Step 2: Count the occurrences of each company using a dictionary
company_counts = {}
for company in flat_list:
    if company in company_counts:
        company_counts[company] += 1
    else:
        company_counts[company] = 1

# Step 3: Print the unique counts of companies
for company, count in company_counts.items():
    print(f"{company}: {count}")

out = pd.DataFrame(company_counts.items(), columns=['Key','Value']).sort_values('Key')
out.to_csv('AAPL_links.csv')    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




