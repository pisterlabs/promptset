
import chromadb
import pandas as pd
from transformers import pipeline, set_seed
from chromadb.utils import embedding_functions
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
import json
import ast

# Define a class named RAG_LLM (Retrieval Augmented Generation using Large Language Model)
class Inference_RAG:
    def __init__(self):
        # Initialize the RAG_LLM class with necessary parameters
        self.openai_api_key = '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'
        self.client = OpenAI(api_key=self.openai_api_key)
        self.chroma_client = chromadb.HttpClient(host='34.93.228.164', port=6969)
        self.collection = self.chroma_client.get_or_create_collection(name="myCollection")

    def delete(self):
        # Method to delete the collection named "myCollection"
        self.chroma_client.delete_collection(name="myCollection")

    def get_embedding(self, text, model="text-embedding-ada-002"):
        # Method to get the embedding of a given text using OpenAI API
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

    def read_data(self, csv_data):
        # Method to read data from a CSV file and add it to the collection
        data = pd.read_csv('/Users/shashank/python/INTER-IIT/Data/mod_data.csv')
        data['documents'] = data['documents'].astype(str)
        data['metadata'] = data['metadata'].apply(ast.literal_eval)
        self.add_data(data)

    def add_data(self, data):
        # Method to add data to the collection
        data_dict_list = data.to_dict(orient='records')
        if 'documents' and 'metadata' in data.columns:
            documents = list(data['documents'])
            m_data = list(data['metadata'])
        else:
            documents = data_dict_list
            m_data = data_dict_list

        embeddings = [self.get_embedding(text) for text in documents]
        ids = [str(self.collection.count() + x) for x in range(len(data_dict_list))]
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=m_data,
            ids=ids
        )

    def get_ids(self, data):
        # Method to get IDs from the collection based on a given query
        inp = self.get_embedding(data)
        return self.collection.query(query_embeddings=[inp], n_results=self.collection.count(), where={'Tool': data})['ids']

    def delete_data(self, data):
        # Method to delete data from the collection based on a given query
        docs = self.get_ids(data)[0]
        print(docs)
        self.collection.delete(ids=docs, where={'Tool': data})
        return "Deleted tool"

    def query(self, data):
        # Method to query the collection based on a given query
        inp = self.get_embedding(data)
        results = self.collection.query(
            query_embeddings=[inp],
            n_results=5
        )
        return results

    def get_json_output(self, user_query):
        # Method to get JSON output from the OpenAI API based on a user query
        inp = f'''
            Query:
            {user_query}

            Here's a Query. Transform the given query into distinct sentences, following the provided examples.ONLY OUTPUT AS JSON :

            Example 1.:
            Query: Prioritize my P0 issues and add them to the current sprint.

            Output:
            {{
            "sentence1": "Address my specific issue.",
            "sentence2": "Emphasize the prioritization of P0 issues.",
            "sentence3": "Add listed issues to the current sprint."
            }}

            Example 2.:
            Query: Summarize high severity tickets from the customer UltimateCustomer

            Output:
            {{
                "sentence1" : "search for customer",
                "sentence2" : "List high severity tickets Coming from 'UltimateCustomer'",
                "sentence2" : "Summarize them"
            }}
        '''

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": inp}
            ]
        )

        # Return the JSON content
        return completion.choices[0].message.content

    def output(self, user_input):
        # Method to process user input, query the collection, and return a formatted output
        Queries = self.get_json_output(user_input)
        json_object = json.loads(Queries)
        Tools_list = ""
        Tools = []
        Args = []

        for key, value in json_object.items():
            doc = ob.query(value)['documents'][0]
            met = ob.query(value)['metadatas'][0]

            for i, j in zip(met, doc):
                if i['Tool'] not in Tools or i['Tool Argument'] not in Args:
                    Tools_list = Tools_list + '\n' + 'Tool : ' + i['Tool'] + '\n' + 'Tool Argument : ' + i[
                        'Tool Argument'] + '\n' + j + '\n'
                    Tools.append(i['Tool'])
                    Args.append(i['Tool Argument'])

        return Tools_list

    def get_prompt(self,query, rag_output):
        prompt=f"""
        ###Set of Tools
        {rag_output}


        ###Query
        {query}

        using the above query and tools list provided, generate a RFC8259 compliant JSON object answering the query.
        """
        return prompt

    def call_fine_tune_gpt(self,query):
        client = OpenAI(api_key=self.openai_api_key)
        rag_output = self.output(query)

        inp = self.get_prompt(query, rag_output)

        response = client.chat.completions.create(
          model="ft:gpt-3.5-turbo-0613:devrev-inter-iit-tech-meet::8UG8JMbv",
          messages=[
            {"role": "user", "content": inp}
          ]
        )
        return response.choices[0].message.content


# Main block to execute the code
if __name__ == "__main__":
    ob = Inference_RAG()
    print(ob.call_fine_tune_gpt('Given a customer meeting transcript T, create action items and add them to my current sprint'))



