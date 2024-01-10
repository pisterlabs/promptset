#TODO: save the logs to a dataframe and then add them to a chroma db to be able to retrieve them easily
import argparse
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import HuggingFaceEmbeddings


from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

#options is a list of strings that are the keys of the json file
def keep_only(json_log, options):
    hits = json_log["hits"]["hits"]
    columns_to_keep = []
    for hit in hits:
        source = hit["_source"]
        keep_only_dict = {}
        for option in options:
            keep_only_dict[option] = source[option]
        columns_to_keep.append(keep_only_dict)
    df = pd.DataFrame(columns_to_keep)
    #fix host column
    try:
        df["host"] = df["host"].apply(lambda x: x['hostname'])
    except:
        pass
    #convert all the columns to string
    df = df.astype(str)
    return df

def create_preprocessed_folder(raw_logs_path):
    preprocessed_logs_path = raw_logs_path.replace("raw", "preprocessed")
    os.makedirs(preprocessed_logs_path, exist_ok=True)
    return preprocessed_logs_path
"""
#embeddings1 and embeddings2 are normalized numpy arrays
def cosine_similarity_vectorized(array_of_embedings_1, array_of_embedings_2):
    #for each row in the array of embeddings calculate the cosine similarity between the two arrays
    #the result is a matrix of cosine similarities
    cosine_similarities = np.dot(array_of_embedings_1, array_of_embedings_2.T)
    return cosine_similarities

def get_compressed_df(df, threshold):
    #if messages are exactly the same and by the same host then drop them and keep only the last one
    #df = df.drop_duplicates(subset=['message', 'host'], keep='last').reset_index()
    #find similar messages and group them using an embedding function and a cosine similarity
    sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    df['embeddings'] = pd.Series(list(sentence_transformer.encode(df['message'].values)))
    #entries for the compressed_df
    list_for_compress_df = []
    #for each host calculate the similarity between the messages
    grouped = df.groupby('host')
    for host, group in tqdm(grouped):
        #sort group by timestamp reverse
        group = group.sort_values(by=['@timestamp'], ascending=False)
        #convert the list in the embeddings column to a numpy array
        embeddings_of_group = np.array([np.array(embedding) for embedding in group['embeddings']])
        #calculate the similarity matrix
        similarity_matrix_in_group_by_message = cosine_similarity_vectorized(embeddings_of_group,embeddings_of_group)
        #go through the group from last message to first message
        #for each message, take all the messages that are similar to it and put them in compressed_df
        already_included = []
        for index_in_group, (df_index, row) in enumerate(group.iterrows()):
            #if the message is already in the compressed_df then skip it
            if index_in_group in already_included:
                continue
            already_included.append(index_in_group)
            #get the similarity matrix row of the message
            similarity_matrix_row = similarity_matrix_in_group_by_message[index_in_group]
            #get the indexes of the messages that are similar to the message
            similar_messages_indexes = np.where(similarity_matrix_row > threshold)[0]
            #remove index from array if is in already_included
            similar_messages_indexes = np.setdiff1d(similar_messages_indexes, already_included)
            #add similar_messages_indexes in already_included
            already_included.extend(similar_messages_indexes)
            #create new entry for the compressed_df
            compressed_log_entry = pd.DataFrame(columns=['host', 'message', '@timestamp','number_of_similar_messages'])
            compressed_log_entry.loc[len(compressed_log_entry.index)] = [row['host'], row['message'], row['@timestamp'], len(similar_messages_indexes)]
            #add the entry to the list
            list_for_compress_df.append(compressed_log_entry)
        if max(already_included) != len(group) - 1:
            print("not matching")
    return pd.concat(list_for_compress_df)
"""
def get_compressed_logs_2(df, threshold):
    df = df.sort_values(by=['@timestamp'], ascending=False)
    sentence_transformer = SentenceTransformer('all-mpnet-base-v2', device='cuda')
    df['embeddings'] = df.apply(lambda x: sentence_transformer.encode(str(x['host']) + str(x['message'])), axis=1)

    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, affinity='cosine', linkage='average')
    clustering_model.fit(df['embeddings'].tolist())
    df['cluster'] = clustering_model.labels_
    #add sum of cluster column
    df['number_of_similar_messages'] = df.groupby('cluster')['cluster'].transform('count')
    #add the host names that are in the cluster
    unique_hosts = df.groupby('cluster')['host'].unique()
    df['hosts_in_cluster'] = df['cluster'].apply(lambda x: unique_hosts[x])
    df = df.drop_duplicates(subset=['cluster'], keep='last').reset_index()
    df = df.drop(columns=['embeddings', 'cluster', 'index'])
    df = df[['hosts_in_cluster', 'number_of_similar_messages', 'message', '@timestamp']]

    return df



def save_json_log_to_df(path_to_json_log):
    options = ["host", "message", "@timestamp"]
    #load the json file
    with open(path_to_json_log) as json_file:
        json_log = json.load(json_file)
    #keep only the message, server and time
    df_keep_only = keep_only(json_log, options)
    #convert the timestamp to datetime if empty logs just ignore
    preprocessed_df_path = path_to_json_log.replace("raw", "preprocessed").replace(".json", ".csv")
    try:
        #convert the timestamp to datetime
        df_keep_only["@timestamp"] = pd.to_datetime(df_keep_only["@timestamp"])
        #sort by timestamp
        df_keep_only = df_keep_only.sort_values(by=['@timestamp'])
    except KeyError:
        df_keep_only.to_csv(preprocessed_df_path, index=False)
        return
    #compressed_df = get_compressed_df(df_keep_only, 0.95)
    compressed_df = get_compressed_logs_2(df_keep_only, 0.05)
    print('compression ratio: ', len(compressed_df)/len(df_keep_only))
    compressed_df.to_csv(preprocessed_df_path, index=False)
    #print size of the df
    print("size of the df: ", len(compressed_df))

def load_csv_as_langchain_docs(path_to_csvs):
    loader = DirectoryLoader(path_to_csvs, glob='**/*.csv', loader_cls=CSVLoader)
    documents = loader.load()
    return documents

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", help="path to the logs folder", required=True)
    args = parser.parse_args()

    #get the list of logs
    logs = os.listdir(args.log_path)
    #create the preprocessed folder
    preprocessed_logs_path = create_preprocessed_folder(args.log_path)

    #for each log
    for log in logs:
        #save the json log to a dataframe
        json_log_path = os.path.join(args.log_path, log)
        save_json_log_to_df(json_log_path)
        #save the dataframe in the preprocessed folder
    

    """
    #load csv as docs
    print(preprocessed_logs_path)
    docs = load_csv_as_langchain_docs(preprocessed_logs_path)
    #docs = [doc for doc in docs if doc!=None]


    db = Chroma.from_documents(docs, def_embedding_function, persist_directory="/data/preprocessed/chromadb")
    db.persist()
    """