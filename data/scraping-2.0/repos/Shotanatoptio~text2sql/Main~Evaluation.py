import streamlit as st
import pandas as pd
import numpy as np
import clickhouse_connect
from dotenv import dotenv_values

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Connect to ClickHouse DB
env_vars = dotenv_values('/root/text2sql/Credentials/.env')
host = env_vars['host']
port = int(env_vars['port'])
username = env_vars['user']
password = env_vars['password']

client = clickhouse_connect.get_client(host=host, port=port, secure=True, username=username, password=password)

# OpenAI
api_key = env_vars['OPENAI_API_KEY']
llm = OpenAI(model_name='gpt-3.5-turbo', openai_api_key=api_key, temperature=0.0)

# Define Context and Additional Contextual Definitions
context_query = """
You are a very experienced data engineer whose job is to write correct SQL queries. Given below is a description of four tables delimited with <>. The descriptions contain information about how tables were created,
what are the columns, their types, definitions, references to other tables using Foreign Keys and finally, first three rows as an example. 

<
Table 1: Users (Information about users, their registration date and activity status)

CREATE TABLE "Users" (
    "UserId" INTEGER NOT NULL (unique identifier of an user),
    "RegDate" DATE NOT NULL (date of registration),
    "Status" NVARCHAR(220) (status of the user: active or passive),
    PRIMARY KEY ("UserId")
)

SELECT * FROM "Users" LIMIT 3;
UserId RegDate Status
120	2023-03-04	passive
345	2023-03-19	active
533	2021-07-24	passive
>

<
Table 2: UserActivity (Information about users visit to the website. It contains history of dates of visitis, channels of visit: direct visit or through clicking an advertisement of marketing campaign. \ 
If visit happened by clicking the ad then corresponding campaign Id is also provided.)

CREATE TABLE "UserActivity" (
    "VisitId" INTEGER NOT NULL (unique identifier of a user's visit to website),
    "UserId" INTEGER NOT NULL (Id of an user),
    "VisitDate" DATE (date of visit),
    "Click" BOOLEAN (if user visited website after clicking an advertisement of marketing campaign on some platform (Google, LinkedIn, Facebook, Bing) then 1, otherwise 0),
    "CampaignId" INTEGER (Id of marketing campaign. If user arrived at website directly without advertisement then CampaignId is 999),
    PRIMARY KEY ("VisitId"),
    FOREIGN KEY("UserId") REFERENCES "Users" ("UserId"),
    FOREIGN KEY("CampaignId") REFERENCES "CampaignActivity" ("CampaignId")
)

SELECT * FROM "UserActivity" LIMIT 3;
VisitId UserId VisitDate Click CampaignId
23	5259	2021-11-27	1	25
24	708	    2023-05-18	1	29
46	7601	2022-11-04	0	7
>

<
Table 3: CampaignActivity (Information about unique marketing campaigns with starting and ending dates, cost of campaign and the platform where the advertisements/campaigns are/were running (LinkedIn, Google, Facebook, Bing))

CREATE TABLE "CampaignActivity" (
    "CampaignId" INTEGER NOT NULL (unique id of marketing campaign),
    "Platform" TEXT NOT NULL (a platform/social media where the advertisement/campaign is/was running),
    "AdStartDate" DATE (start date of advertisement/campaign),
    "AdEndDate" DATE (end date of advertisement/campaign),
    "Cost" REAL (cost of given advertisement/campaign in USD),
    PRIMARY KEY ("CampaignId")
)

SELECT * FROM "CampaignActivity" LIMIT 3;
CampaignId Platform AdStartDate AdEndDate Cost
1	Google	 2022-06-22	 2022-06-27	154.74
2	Facebook 2023-02-14	 2023-03-12	894.79
3	Google	 2022-12-20	 2023-01-18	897.17
>

<
TABLE 4: Customers (Information about clients/customers of marketing agency. Customers are not users. Customers pay money to marketing agency for advertisements/campaigns.)

CREATE TABLE "Customers" (
    "CustomerId" INTEGER NOT NULL (unique identifier of client/customer),
    "Name" TEXT NOT NULL (full name of the customer),
    "Email" TEXT NOT NULL (email of the customer),
    "Status" TEXT NOT NULL (status of the customer: active or passive),
    "CreatedAt" DATAE  (date of account creation/registration),
    PRIMARY KEY ("CustomerId")
)

SELECT * FROM "Customers" LIMIT 3;
CustomerId Name Email Status CreatedAt
36868	Ella Lewis	ella.lewis@example.com	inactive	2022-06-17
49449	Ava Miller	ava.miller@example.com	active	    2021-12-18
50287	Michael Rodriguez	michael.rodriguez@gmail.com	inactive	2022-07-03
>

Carefully analyze tables above and write proper SQL query for the following instructions delimited by triple backticks ```{user_text}```

For the definition of specific terminology you can use following: {def_terminology}

Write query in ClickHouse SQL.

Do not hallucinate. Don't use columns that aren't available in table. Use joins to other tables to find appropriate columns.

result must be just a sql query and nothing else!
"""

def_terminology = """
CPC - Cost per Click (calculated as sum of total cost of advertisement/campaign divided by total number of clicks)
"""

# Evaluation dataset
df_evaluation = pd.read_csv('evaluation_dataset.csv')


# Save app predictions
app_predictions = {}

for user_input in df_evaluation['Text']:
    try:
        # Check Context Before Generating Query
        print("user textual input:", user_input)
        db_context = f"""
        <
        Table 1: Users (Information about users, their registration date and activity status)

        CREATE TABLE "Users" (
            "UserId" INTEGER NOT NULL (unique identifier of an user),
            "RegDate" DATE NOT NULL (date of registration),
            "Status" NVARCHAR(220) (status of the user: active or passive),
            PRIMARY KEY ("UserId")
        )

        SELECT * FROM "Users" LIMIT 3;
        UserId RegDate Status
        120	2023-03-04	passive
        345	2023-03-19	active
        533	2021-07-24	passive
        >

        <
        Table 2: UserActivity (Information about users visit to the website. It contains history of dates of visitis, channels of visit: direct visit or through clicking an advertisement of marketing campaign. \ 
        If visit happened by clicking the ad then corresponding campaign Id is also provided.)

        CREATE TABLE "UserActivity" (
            "VisitId" INTEGER NOT NULL (unique identifier of a user's visit to website),
            "UserId" INTEGER NOT NULL (Id of an user),
            "VisitDate" DATE (date of visit),
            "Click" BOOLEAN (if user visited website after clicking an advertisement of marketing campaign on some platform (Google, LinkedIn, Facebook, Bing) then 1, otherwise 0),
            "CampaignId" INTEGER (Id of marketing campaign. If user arrived at website directly without advertisement then CampaignId is 999),
            PRIMARY KEY ("VisitId"),
            FOREIGN KEY("UserId") REFERENCES "Users" ("UserId"),
            FOREIGN KEY("CampaignId") REFERENCES "CampaignActivity" ("CampaignId")
        )

        SELECT * FROM "UserActivity" LIMIT 3;
        VisitId UserId VisitDate Click CampaignId
        23	5259	2021-11-27	1	25
        24	708	    2023-05-18	1	29
        46	7601	2022-11-04	0	7
        >

        <
        Table 3: CampaignActivity (Information about unique marketing campaigns with starting and ending dates, cost of campaign and the platform where the advertisements/campaigns are/were running (LinkedIn, Google, Facebook, Bing))

        CREATE TABLE "CampaignActivity" (
            "CampaignId" INTEGER NOT NULL (unique id of marketing campaign),
            "Platform" TEXT NOT NULL (a platform/social media where the advertisement/campaign is/was running),
            "AdStartDate" DATE (start date of advertisement/campaign),
            "AdEndDate" DATE (end date of advertisement/campaign),
            "Cost" REAL (cost of given advertisement/campaign in USD),
            PRIMARY KEY ("CampaignId")
        )

        SELECT * FROM "CampaignActivity" LIMIT 3;
        CampaignId Platform AdStartDate AdEndDate Cost
        1	Google	 2022-06-22	 2022-06-27	154.74
        2	Facebook 2023-02-14	 2023-03-12	894.79
        3	Google	 2022-12-20	 2023-01-18	897.17
        >

        <
        TABLE 4: Customers (Information about clients/customers of marketing agency. Customers are not users. Customers pay money to marketing agency for advertisements/campaigns.)

        CREATE TABLE "Customers" (
            "CustomerId" INTEGER NOT NULL (unique identifier of client/customer),
            "Name" TEXT NOT NULL (full name of the customer),
            "Email" TEXT NOT NULL (email of the customer),
            "Status" TEXT NOT NULL (status of the customer: active or passive),
            "CreatedAt" DATAE  (date of account creation/registration),
            PRIMARY KEY ("CustomerId")
        )

        SELECT * FROM "Customers" LIMIT 3;
        CustomerId Name Email Status CreatedAt
        36868	Ella Lewis	ella.lewis@example.com	inactive	2022-06-17
        49449	Ava Miller	ava.miller@example.com	active	    2021-12-18
        50287	Michael Rodriguez	michael.rodriguez@gmail.com	inactive	2022-07-03
        >

        For the definition of specific terminology you can use following: {def_terminology}
        """

        context_checker = """"
        You're the best data engineer in the world. You are an expert in analytics and SQL. Given the database tables described in triple backticks ```{db_context}```, can I write an SQL query using tables mentioned above to answer the following question: {user_input}. Your answer must be Yes or No without comma or dot and nothing more or less!
        """

        prompt_context_checker = PromptTemplate.from_template(context_checker)
        chain_context = LLMChain(llm=llm, prompt=prompt_context_checker)
        result_context_checker = chain_context.run(db_context=db_context, user_input=user_input)
        print("Result Context Checker:", result_context_checker)  # Debug statement


        if result_context_checker == 'Yes':
            # Develop Chain for Correct Query Generation
            prompt_query = PromptTemplate.from_template(context_query)
            chain_query = LLMChain(llm=llm, prompt=prompt_query)
            result_query_init = chain_query.run(user_text=user_input, def_terminology=def_terminology)
            print("Result Query Init:", result_query_init)  # Debug statement


            # Check The Syntax for Generated Query
            sanity_check = """
            You're the best data engineer in the world. You are an expert in SQL. Given the SQL query: {result_query_init}. 
            If the syntax of the query is correct then return original SQL query, otherwise make appropriate changes and return a new query. 
            Your output in both cases must be the only SQL query and nothing else! No text, no comment, no assessment!
            """

            prompt_sanity = PromptTemplate.from_template(sanity_check)
            chain_sanity = LLMChain(llm=llm, prompt=prompt_sanity)
            result_query_final = chain_sanity.run(result_query_init=result_query_init)
            print("Result Query Final:", result_query_final) 

            # Run The Final Query Against Database
            result_from_db = client.command(result_query_final)
            print("Result from DB:", result_from_db)  
            app_predictions[user_input] = result_from_db
    except Exception as e:
        app_predictions[user_input] = e
        print(e)
        pass


# Create prediction dataframe and save it
df_predition = pd.DataFrame(list(app_predictions.items()), columns=['Text', 'Prediction'])
df_predition.to_csv('df_prediction.csv')
print("Dataframe is saved")