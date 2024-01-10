import snowflake.connector 
import snowflake.connector.pandas_tools
import pandas
import ast
import os
import re
from operator import itemgetter

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities import SQLDatabase
from langchain.prompts import BasePromptTemplate, PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

class insuranceBotModel:
    def SQLconversationChain(self):
        for desc in self.table_desc:
            self.SQLmemory.save_context({"input":f"""Below is the table description with its self-descriptive columns and constraints.\nYou will need the below information in order to develop SQL queries so save it in the memory.\nAcknowledge by saying ok \n{desc}"""},
            {"output":"OK"})
        self.SQLconversation = ConversationChain(
            llm=self.SQLmodel,
            verbose=True,
            memory=self.SQLmemory
        )
    def __init__(self, openai_api_key,openai_model, user, password, account, warehouse, database, schema, role):
        # snowflake_url = "snowflake://uddhavz:Thinkpad1234@qh16031.ca-central-1.aws/DB_PROJECT_DAMG7245_TEAM7/stage?warehouse=WH_PROJECT_DAMG7245_TEAM7&role=ACCOUNTADMIN"
        print("\n-----------------------------------------------------------")
        self.openai_api_key = openai_api_key

        try : 
            self.conn = snowflake.connector.connect(
                user=user,
                password=password,
                account=account,
                warehouse=warehouse,
                database=database,
                schema=schema,
                role=role
                )
            
        except Exception as e:
            # return {'answer': f"Database error: {e}",
            #         'followup_questions': []}
            print(f"Error :: {e}")

        print(f"Snowflake database connection Established\n {self.conn}")    
        print(f"insuranceBotModel directory : {os.getcwd()}")
        

        #memory and OpenAI for SQL generation
        self.SQLmodel = ChatOpenAI(temperature=0, model_name = openai_model ,openai_api_key = self.openai_api_key)
        path = os.path.join(os.getcwd(),"frontend/model/table_desc/tableDescription.txt")
        print(f"file path : {path}")
        with open(path, 'r') as file:
            array_str = file.read()
            self.table_desc = ast.literal_eval(array_str)        
        self.SQLmemory = ConversationBufferMemory()
        self.SQLconversationChain()

         #Memory and OpenAI for context generation
        self.chatHistory = """Previous Conversation :\n"""
        self.chatModel = ChatOpenAI(temperature=0, model_name = openai_model ,openai_api_key = self.openai_api_key)
        print("-----------------------------------------------------------\n\n")



    def get_response(self, question, state, county, planid=''):
        validSql = False
        runs = 0
        static_context = f"""
        Generate a runnable SQL query for the question below using the previous information about tables, columns, and constraints. 
        Only use the tables and columns provided earlier. Do not make up any new columns. 
        Before starting with the SQL query go through all the column description provided earlier and then select the suitables columns. 
        One specific area that deserves attention is the referencing of columns in your queries. It's crucial to ensure that columns are sourced only from the tables specified in the FROM clause. perform the necessary joins to fetch columns from other tables.
        The output should be only a SQL query nothing else everytime.
        If the error message is sent then correct the query according to the information provided 
        If a valid query cannot be generated,return NULL.
        Always include INSURANCE_PLAN_ID !
        Use year = 2024 unless the question says to use any other year


            My state = {state} 
            county = {county}
        """
        if planid != '':
            static_context += f"\n\tplanid = {planid}" 
        input_prompt = f"""{static_context} If any state, county or planid is mentioned in question use that information.\n\nQuestion:{question}\n\n Include INSURANCE_PLAN_ID ! LIMIT the results to 100 rows """
        ## SQL Generation
        while(validSql == False):
            
            response = self.SQLconversation.predict(input=input_prompt)

            print(f"\n >> Response:\n\n{response}")

            print("SQL Validation")


            if response[:6].upper() != 'SELECT' or response == 'NULL':
                self.SQLmemory = ConversationBufferMemory()
                self.SQLconversationChain()
                return({'answer': f"Sorry, Due to limited data available. I cannot provide you with an answer :(",
                            'followup_questions': ["Which issuers can provide me Insurance?",
                                                    "Give me 5 plans with cheapest premiums?",
                                                    "List of Plans that cover Dental Insurance."
                                                    ]})

            print("\n\n Checking for database modification queries")
            pattern = re.compile(r"^\s*(drop|alter|truncate|delete|insert|update)\s", re.IGNORECASE)

            if pattern.match(response):
                self.SQLmemory = ConversationBufferMemory()
                self.SQLconversationChain()
                return({'answer': f"Sorry, I can't execute queries that can modify the database.",
                            'followup_questions': []})

            
            ## Data extraction
            print(f"\n\n Retriving Data")
            try:
                cur = self.conn.cursor()
                cur.execute(response)
                validSql = True
            except Exception as e:
                print(f"Error : {e}")
                runs+=1
                if runs < 6:
                    print(f"\n\n Retry No ::{runs}")
                    input_prompt = f"""Error encountered: {e} \n Please choose appropriate column names and tables and correct the query \n Send SQL query ONLY and nothing else """
                    continue
                else:
                    self.SQLmemory = ConversationBufferMemory()
                    self.SQLconversationChain()
                    print(f"ERROR :: {e}")
                    return {'answer': f"Database Connection Error",
                            'followup_questions': []}

        if validSql:         
            df = cur.fetch_pandas_all()
            data = [df.columns.tolist()] + df.values.tolist()

            print(f"\n\n Context generation")
            ## Context generation based on retrived data
            response_schemas = [
                ResponseSchema(name="answer", description="Your answer to the given question",type = 'markdown'),
                ResponseSchema(name="followup_questions", description="A list of 3 insurance related follow-up questions on top the question below.", type = 'list')
            ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()

            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template("""
                As a financial student employed as an insurance agent, you are tasked with analyzing insurance plans and providing responses based on the questions posed to you. The user has extracted relevant data from our insurance dataset to assist in answering the specific question at hand. It is crucial to exclusively utilize this provided information and refrain from incorporating any additional data.
                You have to return 2 things :
                1. A conversational Reponse to the question below using the provided data and previous conversation only. 
                2. A list of three insurance related follow-up questions on top the recent question and the answer you will provide. Do not repeat the suggested questions. 
                    output format : ['Question1', 'Question2','Question3']

                {format_instructions}

                DONT FORGET TO PUT COMMA(,) between to keys in JSON output

                {history}

                Extracted Data:
                {data}

                Question:
                {question} Also include INSURANCE_PLAN_ID if available and neccesary!""")
                ],
                input_variables=["history","data","question"],
                partial_variables={"format_instructions": format_instructions}
            )
            _input = prompt.format_prompt(history = self.chatHistory,data = data, question=question)
            print(f"\n\n{_input}\n\n")
            output = self.chatModel(_input.to_messages())
            print(f"\n\n{output}\n\n")
            
            validJSON = False
            runs = 0
            while (validJSON == False):
                try:
                    json_output = output_parser.parse(output.content)
                    validJSON = True
                except Exception as e :
                    print(f"Error : {e}")
                    runs+=1
                    if runs < 4:
                        print(f"\n\n Retry No ::{runs}")
                        error_prompt = ChatPromptTemplate(
                            messages=[
                                    HumanMessagePromptTemplate.from_template(""" 
                                    Error encountered: {e}

                                    Kindly regenerate the previous answer with proper JSON format
                                    DONT FORGET TO PUT COMMA(,) between to keys in JSON output
                                    {format_instructions}
                                    {history}

                                    Extracted Data:
                                    {data}

                                    Question:
                                    {question} Also include INSURANCE_PLAN_ID if available!""")
                                    ],
                            input_variables=["e","history","data","question"],
                            partial_variables={"format_instructions": format_instructions}
                        )
                        _error_input = error_prompt.format_prompt(e=e,history = self.chatHistory,data = data, question=question)
                        print(f"_error_input:: \n{_error_input}")
                        output = self.chatModel(_error_input.to_messages())
                        print(f"\n\n{output}\n\n")
                    else:
                        return {'answer': f"Something went wrong! Please try again!",
                            'followup_questions': []}
            
            print(f"Context Based Response:\n{json_output}\n\n")
            self.chatHistory += f"""\nUser : {question}
                Insurance Agent : {str(json_output)}"""
            print(f"\n\n Response Sent!")
        return json_output

    def get_county(self,state):
        results = self.conn.cursor().execute(f"SELECT COUNTY from USA_QHP.PUBLIC.STATE_COUNTY where STATE_CODE = '{state}' ORDER BY COUNTY").fetchall()
        countynames = [x[0] for x in results]
        return countynames
    
    def get_states(self):
        results = self.conn.cursor().execute(f"SELECT DISTINCT STATE_CODE from USA_QHP.PUBLIC.STATE_COUNTY ORDER BY STATE_CODE").fetchall()
        countynames = [x[0] for x in results]
        return countynames

    def id_present_in_db(self, state, county, planid):
        results = self.conn.cursor().execute(f"""SELECT 1 as ValidID FROM USA_QHP.PUBLIC.T_YEAR_WISE_INSURANCE_PLAN_DETAILS WHERE state_code_for_insurance_issuer = '{state}' AND county_name_for_insurance_issuer = '{county}' AND insurance_plan_id = '{planid}' LIMIT 1;""").fetchall()
        return not (results == [])



