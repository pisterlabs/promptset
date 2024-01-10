import sys
sys.path.append("../shared_utils")

import os
import json
import time
import requests
from tqdm import tqdm
from openai import OpenAI
from psycopg2 import pool
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from contextlib import contextmanager
from shared_utils.sql_engine import SQLEngine

load_dotenv()

HN_URL = "https://hacker-news.firebaseio.com/v0/"

class Processor:
    def __init__(self) -> None:
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rate_limit_enabled = True
        self.sql_engine = SQLEngine(
            host= os.getenv("POSTGRESQL_URL"),
            database= "postgres",
            user= "postgres",
            password= os.getenv("POSTGRESQL_PWD"),
            minconn= 1,
            maxconn= 5
        )

    def add_pgvector(self) -> None:
        query = \
        """ 
        CREATE EXTENSION vector 
        """
        self.sql_engine.execute_query(query)
    
    def create_hn_embeddings_table(self) -> None:
        query = \
        """
        CREATE TABLE IF NOT EXISTS hn_embeddings (
            id bigserial PRIMARY KEY,
            title varchar(255),
            url varchar(255),
            embedding vector(384),
            hn_post_id bigint,
            created_at timestamp
        )
        """
        self.sql_engine.execute_query(query)
    
    def insert_into_embeddings_table(self, data: Dict) -> None:
        check_query = \
        """
        SELECT * FROM hn_embeddings WHERE hn_post_id = %s

        """
        existed_record = self.sql_engine.execute_select_query(check_query, (data["hn_post_id"],))
        if (len(existed_record) > 0): return

        query = \
        """
        INSERT INTO hn_embeddings(title, url, embedding, hn_post_id, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """
        insert_data = (data["title"], data["url"], data["embedding"], data["hn_post_id"], datetime.now())
        self.sql_engine.execute_insert_query(query, insert_data)
    
    def fetch_and_insert(self, limit: int= 200) -> None:
        self.create_hn_embeddings_table()
        post_ids = self.fetch_top_stories()[:limit]
        for post_id in tqdm(post_ids, desc="fetching and inserting"):
            fetch_result = self.fetch_by_post_id(post_id)
            if not fetch_result: continue
            try:
                transformed_fetch_result = self.transform(fetch_result)
                self.insert_into_embeddings_table(transformed_fetch_result)
            except Exception as e:
                print("inserting failed: ", e)
                continue
            if self.rate_limit_enabled: time.sleep(3)
    
    def transform(self, fetch_result: Dict) -> Dict:
        return {
            "title": fetch_result["title"],
            "url": fetch_result["url"],
            "hn_post_id": fetch_result["id"],
            "embedding": self.get_hf_embeddings(fetch_result["title"]),
            "created_at": datetime.now()
        }
        

    def get_hf_embeddings(self, text: str) -> List[float]:
        HF_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
        headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
        text = text.replace("\n", "")
        resp = requests.post(HF_URL, headers=headers, json={"inputs": text})
        return resp.json()

    def get_embeddings(self, text: str, model: str="text-embedding-ada-002") -> List[float]:
        text = text.replace("\n", "")
        return self.openai_client.embeddings.create(input=[text], model=model).data[0].embedding
    
    def fetch_top_stories(self) -> Dict:
        resp = requests.get(f"{HN_URL}/topstories.json")
        return json.loads(resp.text)
    
    def fetch_by_post_id(self, post_id: int) -> Dict:
        try: 
            resp = requests.get(f"{HN_URL}/item/{post_id}.json")
            return json.loads(resp.text)
        except Exception as e:
            print("fetching HN post failed: ", e)