import json
import openai
import os
from prompts import Prompts
import time
import random


class GptApi:

# region Prompts
    def definePrompts(self, schema: str):

        promptList = [
            {"role": "system", "content": "Your role is to serve as an intermediary between the user and a database, with the primary objective of responding to inquiries using data stored in a SQL Server 2019 database. This entails executing valid queries against the database and interpreting the outcomes to provide answers to the users questions."},
            {"role": "user", "content": "Moving forward, all of your responses will be exclusively in JSON format. When you need to communicate with the user, please adhere to the following format: {\"recipient\": \"user\", \"message\":\"message for the user\"}."},
            {"role": "user",
                "content": "To communicate with the SQL Server, you can utilize the server recipient. When interacting with the server, it is necessary to specify an action. If you intend to query the database, the action should be specified as query. The designated format for executing a query is as follows: {\"recipient\":\"server\", \"action\":\"query\", \"message\":\"SELECT SUM(OrderQty) FROM Sales.SalesOrderDetail;\"}"},
            {"role": "system", "content": "Use the following table schema to create a syntactically correct MS SQL query. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table."},
            {"role": "system", "content": "'" + schema + "'"}
        ]
        return promptList
# endregion

    def __init__(self, api_key: str, api_org: str = "", model: str = "gpt-3.5-turbo", initialPrompt: str = ""):
        if api_org:
            openai.api_key
        openai.api_key = api_key
        self.model = model
        self.initialPrompt = initialPrompt
        
    def retry_with_exponential_backoff(self,prompt:str, model:str):
        retries = 0
        delay = 1

        while retries < 5:  # Maximum number of retries
            try:
                completion = openai.ChatCompletion.create(
                model=model,
                messages=prompt
                )

                # If successful, return the response or extract the desired information
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Error occurred: {e}")

                # Calculate the next delay using exponential backoff formula
                delay = random.uniform(0, delay) * 2

                # Apply jitter to avoid simultaneous retries by multiple clients
                delay = min(delay, 10)

                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

                retries += 1

        # After exhausting all retries, raise an exception or return an error value
        raise Exception("API call failed after maximum retries")


    def getPotentialTables(self, message: str, sender):

        self.initialPrompt.append({"role": "user", "content": message})
        completion = self.retry_with_exponential_backoff(self.initialPrompt, self.model)
        response = completion
        separator = '\n\n'
        response = response.replace("Relevant Table Names:", "")
        response = response.split(separator, 1)[0]
        myTables = response.split(",")
        myTables = [x.strip(' ') for x in myTables]
        return myTables

    def generateResponseSchema(self, message: str, sender: str, schema: str):
        self.messages = self.definePrompts(schema)
        self.messages.append({"role": "user", "content": message})
        completion = self.retry_with_exponential_backoff(self.messages, self.model)
        response = completion
        self.messages.append({"role": "assistant", "content": response})
        return response


