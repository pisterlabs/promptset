import pandas as pd
import time
import sqlite3
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.vectorstores import FAISS
conn=sqlite3.connect('tem.db')

def load_db():
    print("start time:",time.ctime())
    conn=sqlite3.connect('tem.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    tables=[i[0] for i in tables]
    print(tables)
    return tables

from googletrans import Translator
ids=[]
dataframes=[]
meta_data=pd.DataFrame(columns=['Stuff'])

for i in load_db():
    df=pd.read_sql_query("SELECT * FROM "+i,conn)
    
    drop=[i for i in df.columns if 'noname' in i]
    df.drop(drop,axis=1,inplace=True)
    columns=[Translator().translate(i,dest='en').text for i in df.columns]
    df.columns=columns
    print(df.columns)
    df.to_csv("docs/"+i+".csv",index=False,encoding='utf-8-sig')
    df.columns=columns
    if 'Stuff' not in columns:
        continue
    df=df.rename(columns={i:j for i,j in zip(df.columns,columns)})
    df['Stuff']=df['Stuff'].apply(lambda x:x.replace(' ','') if x else x)
    dataframes.append(df)
    ids=ids+df['Stuff'].tolist()
ids=list(set(ids))

meta_data['Stuff']=ids
for i in dataframes:
    meta_data=pd.merge(meta_data,i,on='Stuff',how='left')
# meta_data.rename(columns={'daily trading':'price'},inplace=True)
meta_data.drop_duplicates(subset=['Stuff'],inplace=True,keep='last')
meta_data.dropna(subset=['Road basic address'],inplace=True)

meta_data.to_csv('docs/full.csv',index=False,encoding='utf-8-sig')
meta_data=pd.read_csv('docs/full.csv')
filter_data=meta_data[['Road basic address','Monthly date of the price','Laps on sale','General trading price','Is it a sale','General transaction price','Is the world']]
filter_data.rename(columns={'Road basic address':'Address',"Monthly date of the price":"update Date","General trading price":"price"},inplace=True)
filter_data.dropna(subset=['Address'],inplace=True)
filter_data.to_csv('docs/filter.csv',index=False,encoding='utf-8-sig')


from langchain.document_loaders.csv_loader import CSVLoader
import dotenv
dotenv.load_dotenv()

loader=CSVLoader('docs/filter.csv',encoding='utf-8-sig')
data=loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(data, embeddings)

#save the database
db.save_local('docs/aps.db')