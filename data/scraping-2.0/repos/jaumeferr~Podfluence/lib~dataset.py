import os
import sys

import sys

import openai
import whisper

import numpy as np
import pandas as pd
import nltk
nltk.download('punkt', quiet=True)  # download the NLTK tokenizer

import tiktoken

import pickle
import time
import uuid
from typing import List, Tuple, Set, Dict
import mysql
import mysql.connector

from pytube import YouTube

from engine import EmbeddingEngine, ChatEngine
from gpt3summarizer import GPT3Summarizer

class PodcastDB():
    def __init__(self, 
                 db_name:str, 
                 credentials:Tuple[str,str]=None,
                 host:str="localhost",
                 create:bool=False):
        
        # Default Username
        username = "aldo" if not credentials else credentials[0]
        pw = "testsql123" if not credentials else credentials[1]
        
        # Default table name
        self.table_name = "podcast_details"
        
        # Setup Database
        self.db = mysql.connector.connect(
                  host=host,
                  user=username,
                  password=pw
        )
        print("Established MySQL connection with host.")
        
        ## Create Database
        if create:
            mycursor = self.db.cursor()
            mycursor.execute("SHOW DATABASES")
            for each_db in mycursor:
                if db_name == each_db[0]:
                    print(f"You are creating a database {db_name} that already exists! Please set create = False (default).")
                    return 
            database_sql = f"CREATE DATABASE {db_name}"
            mycursor.execute(database_sql)
            print(f"Created MySQL database {db_name}.")
        
        ## Access Existing Database
        else:
            mycursor = self.db.cursor()
            mycursor.execute("SHOW DATABASES;")
            exists = False
            for each_db in mycursor:
                if db_name == each_db[0]:
                    exists = True
                    break

            if not exists:
                print(f"You are trying to access a database with name: {db_name} that does not exist. Please set create = True.")
                return
        
        ## Connect to Database
        self.db = mysql.connector.connect(
              host=host,
              user=username,
              password=pw,
              database = db_name
            )

        print(f"Established MySQL connection with database {db_name}.")
        
        # Create Database Table
        
        t_cnt = 0
        mycursor = self.db.cursor()
        mycursor.execute("SHOW TABLES;")
        
        ## Check if Table already exists
        for each_name in mycursor:
            if self.table_name == each_name[0]:
                t_cnt += 1

        if not t_cnt:
            table_sql = f"""CREATE TABLE {self.table_name} 
                (pid INT NOT NULL AUTO_INCREMENT PRIMARY KEY, 
                url VARCHAR(255), 
                podcast_title VARCHAR(255), 
                podcast_name VARCHAR(255), 
                transcript_filepath VARCHAR(255), 
                embeddings_filepath VARCHAR(255), 
                list_summary VARCHAR(4096),
                text_summary VARCHAR(4096))"""
            mycursor.execute(table_sql)
            
        print(f"Table {self.table_name} successfully created or are present in database {db_name}.")
        
    def show_db(self, columns : List[str]):
        column_names = ", ".join(columns)
        
        mycursor = self.db.cursor()
        mycursor.execute(f"select {column_names} from {self.table_name}")
        myresult = mycursor.fetchall()
        
        all_entries = []
        for each_entry in myresult:
            all_entries.append(each_entry)
            print(each_entry)
            
        return all_entries
        
    def query_db(self, query : str):
        """
        returns: list of tuples with each tuple representing a row in SQL datatable
        """
        
        mycursor = self.db.cursor()
        mycursor.execute(query)
        myresult = mycursor.fetchall()

        all_entries = []
        for each_entry in myresult:
            all_entries.append(each_entry)
            
        return all_entries
    
    def update_db(self, pid : int, update_dict : Dict[str, str]):
        """
        update_dict : {col_name : new_value} for given pid
        """
        
        all_updates = []
        for key, val in update_dict.items():
            each_update = key + " = " + "'" + val + "'"
            all_updates.append(each_update)
        
        update_str = ", ".join(all_updates)
        where_cond = f"pid = {pid}"
        
        update_query = f"""UPDATE {self.table_name} SET {update_str} WHERE {where_cond}"""
        
        print(update_query)
        mycursor = self.db.cursor()
        mycursor.execute(update_query)
        self.db.commit()
        
        print(f"Record with pid = {pid} has been successfully updated.") 
        
        return
    
    def insert_podcast(self, podcast_params : List[Tuple[str]]):
        """
        insert List of tuples, where each tuple contains podcast_parameters
                (url VARCHAR(255), 
                podcast_title VARCHAR(255), 
                podcast_name VARCHAR(255), 
                transcript_filepath VARCHAR(255), 
                embeddings_filepath VARCHAR(255), 
                list_summary VARCHAR(4096),
                text_summary VARCHAR(4096))
        
        """
        mycursor = self.db.cursor()
        
        sql = f"""INSERT INTO {self.table_name} (url, podcast_title, podcast_name, transcript_filepath, embeddings_filepath,
                        list_summary, text_summary) VALUES (%s, %s, %s, %s, %s, %s, %s)"""

        mycursor.executemany(sql, podcast_params)

        self.db.commit()
        print(f"{mycursor.rowcount} records successfully inserted.")

        
class PodcastDataset():
    def __init__(self, 
                 base_dir="podcast_files/", 
                 dataset_type="db",
                 database:PodcastDB=None, 
                 df_path=None
                ):
        
        # Setup Directories
        self.base_dir = base_dir
        self.audio_dir = self.base_dir + "audio/"
        self.transcript_dir = self.base_dir + "transcript/"
        self.embeddings_dir = self.base_dir + "embeddings/"
        self.sub_dirs = [self.audio_dir, self.transcript_dir, self.embeddings_dir]
        
        self.init_dir()
        
        # Setup Dataset
        self.dataset_type = dataset_type
        if dataset_type == "df":
            self.df_path = df_path
            if self.df_path is None:
                self.init_dataset_df()
            else: 
                self.load_dataset_df()
        elif dataset_type == "db":
            if database is None:
                raise ValueError("Expected database of type PodcastDatabase, got None instead.")
            self.db_handler = database
    
    def init_dir(self):
        os.makedirs(self.base_dir, exist_ok=True)
        for sub_dir in self.sub_dirs:
            os.makedirs(sub_dir, exist_ok=True)
    
    def init_dataset_df(self, df_name="default.csv"):
        # Define the columns for the DataFrame
        self.dataset_columns = ["pid", "url", "podcast_title", "podcast_name", 
                                "transcript_filepath", "embeddings_filepath", "list_summary", "text_summary"]

        # Create an empty DataFrame with the specified columns
        df = pd.DataFrame(columns=self.dataset_columns)

        # Save the DataFrame to a CSV file
        df.to_csv(df_path, index=False)
        
        self.dataset_df = df
        self.df_path = df_path + df_name
    
    def load_dataset_df(self):
        self.dataset_df = pd.read_csv(self.df_path)
        self.dataset_columns = self.dataset_df.columns.tolist()
    
    def get_podcasts(self, pid=None, podcast_name=None, select_all = False):
        if select_all:
            query = f"SELECT * from {self.db_handler.table_name}"
        else:
            if pid is not None:
                if podcast_name is not None:
                    query = f'''SELECT * from {self.db_handler.table_name} 
                        WHERE pid={pid} AND podcast_name="{podcast_name}"'''
                else:
                    query = f'''SELECT * from {self.db_handler.table_name} 
                        WHERE pid={pid}'''
            else:
                if podcast_name is not None:
                    query = f'''SELECT * from {self.db_handler.table_name} 
                        WHERE podcast_name="{podcast_name}"'''
                else:
                    print("Expected either pid or podcast_name to have a value, instead got None for both")
                    return -1
        
        podcasts = self.db_handler.query_db(query)
        
        podcasts_dict = {}
        for podcast in podcasts:
            podcast_dict = {}
            podcast_dict['pid'] = podcast[0]
            podcast_dict['url'] = podcast[1]
            podcast_dict['podcast_title'] = podcast[2]
            podcast_dict['podcast_name'] = podcast[3]
            podcast_dict['transcript_filepath'] = podcast[4]
            podcast_dict['embeddings_filepath'] = podcast[5]
            podcast_dict['list_summary'] = podcast[6]
            podcast_dict['text_summary'] = podcast[7]
            
            podcasts_dict[podcast[0]] = podcast_dict
            
        return podcasts_dict
    
    def add_podcast_from_list(self, podcast_params : List[Tuple[str]]):
        if self.dataset_type == "db":
            print("\nSaving Dataset to DB...")
            self.db_handler.insert_podcast(podcast_params)
        
        elif self.dataset_type == "df":
            new_rows = [{"pid": podcast_params[0], 
                         "url": podcast_params[1], 
                         "podcast_title": podcast_params[2], 
                         "podcast_name": podcast_params[3], 
                         "transcript_filepath": podcast_params[4], 
                         "embeddings_filepath": podcast_params[5], 
                         "list_summary": podcast_params[6],
                         "text_summary": podcast_params[7]}]

            new_df = pd.DataFrame(new_rows)
            self.dataset_df = pd.concat([self.dataset_df, new_df], ignore_index=True)

            print("\nSaving Dataset to DF...")
            self.dataset_df.to_csv(self.df_path, index=False)

    def add_podcast_from_url(self, url, podcast_name="", openai_key="", delete_audio=True):
        podcast_uuid = str(uuid.uuid4())

        print("\nDownloading Audio...")
        audio_filepath, podcast_title = self.download_audio_from_youtube(url, podcast_uuid)
        
        print("Processing Transcript...")
        raw_transcript, transcript_df, transcript_filepath = self.transcribe_audio(audio_filepath, podcast_uuid)
        
        print("Generating Embeddings...")
        embeddings_filepath = self.gen_embeddings(transcript_df, podcast_uuid, openai_key=openai_key)
        
        print("Generating Summary...")
        list_summary, text_summary = self.gen_summary(raw_transcript, podcast_uuid, openai_key=openai_key)
        
        if delete_audio:
            os.remove(audio_filepath)
        
        podcast_params = [(url, podcast_title, podcast_name,
                          transcript_filepath, embeddings_filepath, 
                          list_summary, text_summary)]
            
        self.add_podcast_from_list(podcast_params)
    
    def download_audio_from_youtube(self, url, podcast_id):
        youtube_video = YouTube(url)
        audio_stream_set = youtube_video.streams.filter(only_audio = True)
        audio_stream = audio_stream_set.first() # Select quality audio stream

        audio_filename = self.audio_dir + str(podcast_id) + ".mp4"
        try:
            audio_stream.download(filename = audio_filename) # Download video
        except Exception as e:
            print(e)
            sys.exit(0)
            
        return audio_filename, youtube_video.title
    
    def transcribe_audio(self, audio_filename, podcast_id):
        whisper_model = whisper.load_model('base')
        
        # Transcribe using Model
        output = whisper.transcribe(model= whisper_model, audio=audio_filename , fp16 = False) # Get transcript
        
        # Tokenize and save as csv file
        transcript = output['text']

        # create a Pandas DataFrame with one row for each sentence
        trans_df = pd.DataFrame({'content': nltk.sent_tokenize(transcript)})

        # add a new column with the length of each sentence
        trans_df['token'] = trans_df['content'].apply(len)
        trans_df = trans_df.reset_index()
        trans_df = trans_df[['index', 'content', 'token']]

        # save the DataFrame to a CSV file
        transcript_filename = self.transcript_dir + str(podcast_id) + "_transcript.csv"
        trans_df.to_csv(transcript_filename, index=False)
        
        return transcript, trans_df, transcript_filename
    
    def gen_embeddings(self, transcript_df, podcast_id, openai_key=""):
        embed_engine = EmbeddingEngine(openai_key)
        
        embeddings = embed_engine.compute_doc_embeddings(transcript_df, label="content")
        embeddings_filename = self.embeddings_dir + str(podcast_id) + "_embedding.pickle"
        
        # Save embeddings as pickle
        with open(embeddings_filename, 'wb') as f:
            pickle.dump(embeddings, f, protocol= pickle.HIGHEST_PROTOCOL)
        
        return embeddings_filename
    
    def gen_summary(self, raw_transcript, podcast_id, openai_key="", max_sentences=10):
        # Generate List Summary and Full Summary
        summarizer = GPT3Summarizer(openai_key, model_engine="gpt-3.5-turbo")
        list_summary, full_summary = summarizer.summarize(raw_transcript, max_sentences)
        
        # Generate Text Summary
        text_summary_prompt = f"Instructions:\nSummarize the given Context."
        text_summary_prompt += f"Limit your answer to a paragraph of {max_sentences} sentences. "
        text_summary_prompt += f"\n\nContext: {full_summary}"
        text_summary_prompt += f"\n\nSummary: "
        chat_engine = ChatEngine(openai_key=openai_key)
        chat_engine.prompt = text_summary_prompt
        text_summary = chat_engine.call()

        return list_summary, text_summary

    
class PodcastDataDF():
    def __init__(self, dataset:PodcastDataset, podcast_id):
        self.dataset = dataset.dataset_df
        self.id = podcast_id
        
        self.load_data()
    
    def load_data(self):
        mask = self.dataset["id"] == self.id
        matching_row = self.dataset[mask]
        self.data = matching_row
        
    def show_data(self):
        print(self.data)
        
    def get_transcription(self):
        return pd.read_csv(self.data['transcript_filepath'].values[0])
    
    def get_embeddings(self):
        with open(self.data['embeddings_filepath'].values[0], 'rb') as f:
            return pickle.load(f)
    
    def get_summary(self):
        return self.data['summary'].values[0]
    
    
