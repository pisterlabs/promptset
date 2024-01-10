from app.controllers.retrieval.interface import QueryEngine
from typing import Optional
import json
from wasabi import msg
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

class RetrieverEngine(QueryEngine):

    def generate_response(self, messages):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            # stream=True
        )
        return response
    
    def query(self, query_string: str) -> tuple:
        query_results = (
            QueryEngine.client.query.get(
                class_name="Chunk",
                properties=["text", "doc_name", "chunk_id", "doc_uuid", "doc_type"],
            )
            .with_hybrid(query=query_string)
            .with_generate(
                grouped_task=f"You are a RAG chatbot, answer the query {query_string} using only the information provided in the context, and be sure to answer in Chinese"
            )
            # .with_additional(properties=["score"])
            .with_limit(1)
            .do()
        )

        if "data" not in query_results:
            raise Exception(query_results)

        results = query_results["data"]["Get"]["Chunk"]
        openai_res  = self.generate_response(query_string)
        return (results, openai_res)
    
    def openai_query(self, query_string, chat_history)-> tuple:

        # Get the current Query
        last_content = query_string[-1]['content'] if query_string else None
        print("\n ==Current query question===",last_content)
        # Query through the vector database
        vector_query_results = (
            QueryEngine.client.query.get(
                class_name="Chunk",
                properties=["text", "doc_name", "chunk_id", "doc_uuid", "doc_type"],
            )
            .with_hybrid(query=last_content)
            # .with_additional(properties=["score"])
            .with_additional("score")
            .with_limit(1)
            .do()
        )

        # Results from the vector database query
        vector_res = vector_query_results["data"]["Get"]["Chunk"]
        print("\n ==Vector database query results===",vector_query_results)
        print("\n ==Results from vector database===",vector_res)
        
        pre_query = f"{last_content}"

        for item in vector_res:
            # Ensure '_additional' dictionary exists and contains 'score' key
            if '_additional' in item and 'score' in item['_additional']:
                # Convert the score value to a float
                score = float(item['_additional']['score'])

                print("\n ====score====",score)
                if score > 0.0121:
                    pre_query = f"You are a RAG chatbot, answer the query {last_content} using only the context provided in {vector_res}, and be sure to answer in Chinese"
                    print("\n ====pre_query====", pre_query)
                    break  # Exit the loop if no need to check other items
        
        print("\n ====Final pre_query====", pre_query)  # Check the value of pre_query after the loop
        database_results = [
            {'role': 'user', 'content': pre_query}
        ]
        transformed_initial_data = [{'role': role, 'content': content} for role, content in chat_history]

        # Remove the last object
        transformed_initial_data.pop()

        # Array to be merged
        database_results = [
            {'role': 'user', 'content': pre_query}  # Assuming pre_query is already defined
        ]

        # Merge the two lists
        merged_data = transformed_initial_data + database_results

        print("\n ===database_results===",database_results)
        print("\n ===chat_history===",chat_history)
        print("\n ===transformed_initial_data===",transformed_initial_data)
        print("\n ===merged_data===",merged_data)
        
        # Query OpenAI with the results from the vector database
        openai_res = self.generate_response(merged_data)
        print("\n ==Results from OpenAI query===",openai_res)
        return openai_res
    
    def api_openai_query(self, query_string, chat_history)-> tuple:

        # Get the current Query
        last_content = query_string[-1]['content'] if query_string else None
        print("\n ==Current query question===",last_content)
        # Query through the vector database
        vector_query_results = (
            QueryEngine.client.query.get(
                class_name="Chunk",
                properties=["text", "doc_name", "chunk_id", "doc_uuid", "doc_type"],
            )
            .with_hybrid(query=last_content)
            # .with_additional(properties=["score"])
            .with_additional("score")
            .with_limit(1)
            .do()
        )

        # Results from the vector database query
        vector_res = vector_query_results["data"]["Get"]["Chunk"]
        print("\n ==Vector database query results===",vector_query_results)
        print("\n ==Results from vector database===",vector_res)
        
        pre_query = f"{last_content}"

        for item in vector_res:
            # Ensure '_additional' dictionary exists and contains 'score' key
            if '_additional' in item and 'score' in item['_additional']:
                # Convert the score value to a float
                score = float(item['_additional']['score'])

                print("\n ====score====",score)
                if score > 0.0121:
                    pre_query = f"You are a RAG chatbot, answer the query {last_content} using only the context provided in {vector_res}, and be sure to answer in Chinese"
                    print("\n ====pre_query====", pre_query)
                    break  # Exit the loop if no need to check other items
        
        print("\n ====Final pre_query====", pre_query)  # Check the value of pre_query after the loop
        database_results = [
            {'role': 'user', 'content': pre_query}
        ]
        transformed_initial_data = [{'role': role, 'content': content} for role, content in chat_history]

        # Remove the last object
        transformed_initial_data.pop()

        # Array to be merged
        database_results = [
            {'role': 'user', 'content': pre_query}  # Assuming pre_query is already defined
        ]

        # Merge the two lists
        merged_data = transformed_initial_data + database_results

        print("\n ===database_results===",database_results)
        print("\n ===chat_history===",chat_history)
        print("\n ===transformed_initial_data===",transformed_initial_data)
        print("\n ===merged_data===",merged_data)
        
        # Query OpenAI with the results from the vector database
        openai_res = self.generate_response(merged_data)
        print("\n ==Results from OpenAI query===",openai_res)
        # return openai_res.choices[0].message.content
        return openai_res

    def retrieve_document(self, doc_id: str) -> dict:

        document = QueryEngine.client.data_object.get_by_id(
            doc_id,
            class_name="Document",
        )
        return document

    def retrieve_all_documents(self) -> list:
        query_results = (
            QueryEngine.client.query.get(
                class_name="Document", properties=["doc_name", "doc_type", "doc_link"]
            )
            .with_additional(properties=["id"])
            .with_limit(1000)
            .do()
        )

        print(query_results,"=====")
        results = query_results["data"]["Get"]["Document"]
        return results

    def search_documents(self, query: str) -> list:
        query_results = (
            QueryEngine.client.query.get(
                class_name="Document", properties=["doc_name", "doc_type", "doc_link"]
            )
            .with_bm25(query, properties=["doc_name"])
            .with_additional(properties=["id"])
            .with_limit(20)
            .do()
        )
        results = query_results["data"]["Get"]["Document"]
        return results
