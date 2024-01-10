from django.db import models
import os
import openai
import lancedb
import pandas as pd
import pyarrow as pa
from langchain.embeddings import OpenAIEmbeddings

class QABot:
    def __init__(self):
        self.init_openai()
        self.init_database()

    def init_openai(self):
        os.environ["OPENAI_API_KEY"] = "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"
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "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"
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def init_database(self):
        self.ID_COLUMN_NAME = "id"
        self.VECTOR_COLUMN_NAME = "vector"
        self.QUESTION_COLUMN_NAME = "question"
        self.ANSWER_COLUMN_NAME = "answer"
        self.table_id_counter = 0
        self.embeddings = OpenAIEmbeddings()
        self.table = self.create_table()

    def create_table(self, name="chat", mode="overwrite"):
        db = lancedb.connect("data/lance-cache")
        vector_size = 1536
        schema = pa.schema([
            pa.field(self.ID_COLUMN_NAME, pa.int64()),
            pa.field(self.VECTOR_COLUMN_NAME, lancedb.vector(vector_size)),
            pa.field(self.QUESTION_COLUMN_NAME, pa.string()),
            pa.field(self.ANSWER_COLUMN_NAME, pa.string())])
        table = db.create_table(name, schema=schema, mode=mode)
        return table

    def add_new(self, embeding, question, answer=None):
        self.table.add([{self.ID_COLUMN_NAME: self.table_id_counter, self.VECTOR_COLUMN_NAME: embeding, self.QUESTION_COLUMN_NAME: question, self.ANSWER_COLUMN_NAME: answer}])
        self.table_id_counter += 1

    def delete(self, id):
        query = self.ID_COLUMN_NAME + " = " + id
        self.table.delete(query)

    def search(self, embeding, limit=1, distance=0.15):
        if not self.table.to_pandas().empty:
            df = self.table.search(embeding).limit(limit).to_df()
            if not df.empty and df['_distance'][0] <= distance:
                print("Found with distance of " + str(df['_distance'][0]))
                return df
        return None

    def llm_prompt(self, question, engine="text-davinci-003", max_tokens=50):
        response = openai.Completion.create(
            engine=engine,
            prompt=question,
            max_tokens=max_tokens
        )
        answer = response.choices[0].text.strip()
        return answer

    def get_answer(self, question):
        embeding = self.embeddings.embed_query(question)
        result = self.search(embeding)
        if result is not None:
            print(result)
            print("Search Result from Cache")
            return result[self.ANSWER_COLUMN_NAME].values[0]
        else:
            answer = self.llm_prompt(question)
            self.add_new(embeding, question, answer)
            print(question)
            print("Search Result from LLM")
            return answer
    

# Define a Django model that incorporates the QABot class
class QABotCache(models.Model):
    bot = QABot()

    class Meta:
        abstract = True  # So this model doesn't create a physical database table

    @staticmethod
    def get_answer(question):
        return QABotCache.bot.get_answer(question)
