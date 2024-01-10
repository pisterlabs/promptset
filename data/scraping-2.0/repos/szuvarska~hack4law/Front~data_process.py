import pandas as pd
import re

from pathlib import Path

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document


# Potrzebne pakiety to: 'langchain[chromadb], sentence-transformer i chromadb


def return_df_with_similarities(query:str, tags = ['Kary pieniężne', 'Wyrównanie szkody'], data_path = '../Data/output_for_frontend_2.csv' ):

    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    chroma_db_filepath = Path("./chroma_db")

    df = pd.read_csv(data_path, sep = ";")

    if chroma_db_filepath.exists():
        db = Chroma(embedding_function=embeddings, persist_directory=str(chroma_db_filepath))
    else:
        documents = df.apply(lambda row: Document(page_content = row['uzasadnienie']
                                                , metadata = {'source' : row['id']}), axis=1)

        text_splitter = CharacterTextSplitter(chunk_size=3000 , chunk_overlap=300)
        docs = text_splitter.split_documents(documents)
        db = Chroma.from_documents(docs, embeddings, persist_directory=str(chroma_db_filepath))

    sim = db.similarity_search_with_score(
        query, k=len(df)
        
    )

    results = [(score, doc.metadata["source"], doc.page_content) for (doc, score) in sim]
    results.sort(key=lambda x: x[0])

    # pprint.pprint(results)
    
    df_results = pd.DataFrame(results, columns = ['similarity', 'id', 'uzasadnienie'])
    df_results['id'] = df_results['id'].astype('str')
    df['id'] = df['id'].astype('str')
    df = df[[col for col in df.columns if col!="uzasadnienie"]]
    merged = df_results.merge(df, on='id', how='inner')

    condition = merged['tags'].apply(lambda row_tags: all(tag in row_tags for tag in tags))
    filtered_merged = merged.loc[condition]
    filtered_merged = filtered_merged.sort_values(by=['similarity'], ascending=True).iloc[:10, :]
    return filtered_merged


df = pd.read_csv("Front/clean_output1.csv",sep = ';')
def get_judges(df):
    pattern = r"'name': '(.*?)',"
    for i in range (df.shape[0]):
        df['judges'][i] = re.findall(pattern, df['judges'][i])

    return df

def get_caseNumber(df):
    pattern = r"'caseNumber': '(.*?)'"
    for i in range (df.shape[0]):
        df['courtCases'][i] = re.findall(pattern, df['courtCases'][i])
    return df

df = pd.read_csv('../Data/clean_output2')
df_r = get_judges(df)
df_r = get_caseNumber(df_r)

