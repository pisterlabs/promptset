import requests
import time
import datetime
import openai
import os
import argparse
import sys

os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")
MAX_TOKEN=2000

class Utility:
    def __init__(self):
        pass
    
    def get_current_time(self):
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d_%H:%M:%S")
        return formatted_datetime
util = Utility()

##############################################################################################
# Information about all tables must be stored in a single 'config' folder (see examples)
# That folder must have one folder per table with folder name same as the table name
# Each table folder must have three files:
# 1. platform.txt: One line containing the platform table is on (mysql, mongodb)
# 2. admin.txt: Has all necessary info to establish connection to the data platform
#               One spec per line as  "word_description : value" eg: "database name : db1"
# Note: It is important for admin.txt to be exhaustive, else chatGPT may introduce its own
#       variable names, which can't be progrmatically identified and the python code given
#       by it will fail to execute
#       # TODO: Add port option (URI in gneral)
# 3. schema.txt: Schema of the equivalent SQL table. Each line is a column information in
#                "COLUMN_NAME  type  NULL" format
#    # TODO: How to specify non-SQL structures like lists etc?
#    # TODO: Support for simple files, like json etc
#    # TODO: Add table name to schema as well, else have to see both admin and schema to contruct queries
#    # TODO: Many tables may have common admin file
#
# Once these are defined, feed it into this program by calling the 
# Datalake.define_datalake_from_folder(...) function.
#
# Then to create prompt based on that config, use the Prompt.get_full_query(...)
# Queries can be fed in through files ('query/query1.txt here). And outputs can saved
# in files ('query_output.query1.txt' here) 
##############################################################################################

''' 
Example Prompt:

- I have a mysql table named DETAILS and a mongoldb table named INTERESTS.
- DETAILS has the columns NAME, AGE, ADDRESS. INTERESTS has the fields of NAME, INTEREST. INTEREST is a list of strings and can be empty.
- But the user of my data thinks all data is stored in mysql.
- They wrote the following query:
- SELECT DETAILS.AGE, INTERESTS.INTEREST
- FROM DETAILS JOIN  INTERESTS ON DETAILS.NAME = INTERESTS.NAME
- Please generate a python code to execute this query. My mysql password is my-pwd. Output of the query must be written to a file name query_output.csv
'''

###################################################################################

class SQL_COLUMN:
    def __init__(self, column_name, datatype, null_property="NO_NULL"):
        self.name = column_name
        self.datatype = datatype
        self.null_property = null_property

# SQL schema of a table consists of
# 1. Primary key of the table
# 2. Columns: Each column has a name, type of data it hold, and propoerties like 'can it be NULL?' 
class SQL_SCHEMA:
    def __init__(self):
        self.primary_key_name = None
        self.columns          = {}                # column_name:SQL_COLUMN including primary key
        self.no_columns       = len(self.columns)

    def add_column(self, Column): # input: SQL_COLUMN object
        self.columns[Column.name] = Column
        self.no_columns       = len(self.columns)

    def print_schema(self):
        print("Primary Key is: ", self.primary_key_name)
        print ("Number of columns: ", self.no_columns)
        print("Schema: (Column_Name   Type   Null_property)")
        for _, col in self.columns.items():
            print(f" {col.name}  {col.datatype}  {col.null_property}")
        print("******************************************")


class SQL_TABLE:
    def __init__(self, table_name, schema_filepath=""):
        self.name = table_name
        self.schema = SQL_SCHEMA()  
        self.define_schema_from_file(schema_filepath) 

    # self.schema is populate from a schema file of the format:
    # COLUMN_NAME1 type NULL
    # COLUMN_NAME2 type NO_NULL ....  
    def define_schema_from_file(self, filepath):
        ## populate self.schema
        if filepath == "" :
            print("SQL Table {}: No Schema filename given".format(self.name))
            return 0
        
        file1 = open(filepath, 'r')
        lines = file1.readlines()
        line_no = 0
        
        for line in lines:
            line_no += 1
            col_spec = line.strip().split()   # Column_name Cloumn_type null_or_nonull
            if line_no == 1: # primary key
                self.schema.primary_key_name = col_spec[0]
            column = SQL_COLUMN(col_spec[0], col_spec[1], col_spec[2])
            self.schema.add_column(column)
        
        self.print_table()
        return 0


    def print_table(self):
        print("Table name is: ", self.name)
        self.schema.print_schema()
        


####################################################################################
# * Table class represents a generic table in any of the data platforms like mysql, mongodb etc
# * Properties common to tables in all platforms (like the need for some admin details) is 
#   to be defined in this class
# * If we later need to specialize for data platforms, child classes of Table can be defined
# * Each table needs 2 files: admin details file, equivalent sql schema file
# TODO: May need a separate platform specific schema file too

class Table:
    def __init__(self, table_name, platform, admin_file="", schema_file=""):
        self.platform       = platform                                        # mysql, mongodb etc
        self.name           = table_name  
        self.admin_details  = {}                                              # Keys must be strings describing what the values are and
                                                                              # -- the variable names for those you want in the code,
                                                                              # -- This will be used to tell chatgpt about admin details
                                                                              # -- eg: "name" : db_name, "password" : sql_pwd etc
        self.equivalent_sql_table = SQL_TABLE(table_name, schema_file)                    # A table, no matter which platform, should have an SQL table equivalent,
                                                                              # -- which is how it is presented to the buisness analyst (the user). 
        
        self.special_case = None                                              # any special rules like how to handle nulls etc

        if self.platform == "mysql":                                          
            self.column_equivalent = "columns"                                 # what are "columns" called in that platform   
        elif self.platform == "mongodb":
            self.column_equivalent = "fields"     
        else:
            sys.exit("Invalid platform name. Should be: mysql, mongodb")      # TODO: move this kind of information to platform.txt

        self.define_admin_from_file(admin_file)


    def define_admin_from_file(self, filepath):
        ## populate self.admin_details
        if filepath == "" :
            print("Table {}: No admin filename given".format(self.name))
            return 0
        print(f"Updating admin info for table {self.name} from {filepath} :")
        file1 = open(filepath, 'r')
        lines = file1.readlines()
        
        for line in lines:
            spec = line.split(":")   # "spec description" : "spec value"
            self.admin_details[spec[0].strip()] = spec[1].strip()
            
        print("Admin info update complete!")
        self.print_admin_info()
        print("---------------------------------------")
        return 0
    
    def print_admin_info(self):
        print(f"Table {self.name} admin info :")
        for k, v in self.admin_details.items():
            print(f"  {k} : {v}")
            


##################################################################################################

# Collection of all Table objects in my data setup

class Datalake:
    def __init__(self, name):
        self.name = name
        self.tables = {}        # table_name : Table
        self.no_tables  = len(self.tables)

    def add_table(self, table): # table is a Table object
        self.tables[table.name] = table
        self.no_tables  = len(self.tables)

    # Folder should contain one folder per table in the datalake
    # Each table folder should contain: platform.txt, schema.txt, admin.txt
    def define_datalake_from_folder(self, folder_path):
        table_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        for tab in table_folders:
            table_name = tab.split('/')[-1]
            platform_file = tab+'/platform.txt'
            admin_file    = tab+'/admin.txt'
            schema_file   = tab+'/schema.txt'
            with open(platform_file) as f:
                platform = f.readline()
            self.add_table(Table(table_name, platform, admin_file, schema_file))
            print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        print("Table definitions complete! \n\n")
            
###################################################################################################

####################################### UTILS ####################################################
## Either reurns ", " or " and " or "." depending on which element you are
## appending to a sentence
## eg: "table1, table2 and table3.""
def add_delimiter(i, L):
    if i < (L-1):
        return f", "
    elif i == (L-1):
        return f" and "
    else:
        return f". "

## query can be passed as a string or as a filepath (with isfile=True)
def get_query(query, isfile=False):
    if isfile:
        with open(query, 'r') as file:
            q = file.read()
        return q
    else:
        return query

#############################################################################
# Prompt for a given Datalake setup
class Prompt:
    def __init__(self, datalake, output_file="query_output.csv"):
        self.datalake      = datalake
        self.output_file   = output_file

        # prefaces
        self.datalake_info_pref = "I have organized my data as follows: "   ## <table1> in <db1>, ....
        self.schema_info_pref   = ""                                        ## "<table1> has <columns/fields etc> <cloumn_name>, of type <type>, and can be <special_cases>"
        self.admin_info_pref    = "Details of my databases are as follows : "
        self.story              = "But the user of my data thinks all the data is stored in mysql with the same column names."  # TODO: "with same column names"?
        # self.sql_schema_pref    = "They think the tables have the following schemas : "       # TODO: Is this  required
        self.query_pref         = "With that assumption, they wrote the following query: "
        self.output_spec        = f"Generate a python code to execute this query on my original data. Query's output should be written to the file {self.output_file}. Please output only the python code and a bash command to installtion all dependencies to run that python code."

        self.datalake_info = ""
        self.schema_info   = ""
        self.admin_info    = ""

    def to_dict(self):
        member_variables = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}
        for k, v in member_variables.items():
            print(f"* {k}: {v}")
        return member_variables
    
    def gen_datalake_info(self):
        ## <table1> in <db1>, ....
        temp = ""

        for index, table_n in enumerate(self.datalake.tables):
            tab = self.datalake.tables[table_n]
            temp += f"table {table_n} in {tab.platform}"
            temp += f"{add_delimiter(index+1, self.datalake.no_tables)}"

        self.datalake_info = self.datalake_info_pref + temp
        return self.datalake_info
    
    def gen_schema_info(self):
        ## "<table1> has <columns/fields etc> <cloumn_name>, of type <type>, and can be <special_cases>"
        temp = ""
        for _, table_n in enumerate(self.datalake.tables):
            tab = self.datalake.tables[table_n]                  # Table object 
            temp += f"Table {table_n} has the following {tab.column_equivalent}: "

            for i, col_name in enumerate(tab.equivalent_sql_table.schema.columns):
                col = tab.equivalent_sql_table.schema.columns[col_name]   # SQL_COLUMN object
                temp += f"'{col_name}' of type {col.datatype}"
                temp += f"{add_delimiter(i+1, tab.equivalent_sql_table.schema.no_columns)}"

        self.schema_info = self.schema_info_pref + temp
        return self.schema_info
    
    def gen_admin_info(self):
        ## "for <table1> the <hostname> is <insert>, the <password> is <inser> "
        temp = ""
        for _, table_n in enumerate(self.datalake.tables):
            tab = self.datalake.tables[table_n]                  # Table object 
            temp += f" For table {table_n} "

            L = len(tab.admin_details)
            for i, spec_name in enumerate(tab.admin_details):
                temp += f"the {spec_name} is {tab.admin_details[spec_name]}"
                temp += f"{add_delimiter(i+1, L)}"

        self.admin_info = self.admin_info_pref + temp
        return self.admin_info
    
    
    def gen_full_prompt(self, query, isfile=False):
        q = get_query(query, isfile)
        self.gen_datalake_info()
        self.gen_schema_info()
        self.gen_admin_info()

        prompt = ""
        prompt +=  self.datalake_info + '\n' + self.schema_info + "\n" + self.admin_info + " \n" + self.story + "\n" + self.query_pref + q + "\n" + self.output_spec
        
        print("The final prompt is: \n\n")
        print(prompt)
        return prompt

#####################################################################################################

class Multi_Message_ChatGPTQuery:
    def __init__(self):
        self.messages = list()
        self.input_message_len = list()
        self.data = ""
        self.runtime = -1
        self.output_text = ""
        self.gpt_model = "gpt-4" # "gpt-3.5-turbo"
        self.finished_reason = ""
        self.response = ""
        self.created_time = -1
        self.uid = ""
        self.completion_tokens = -1
        self.prompt_tokens = -1
        self.total_tokens = -1
        
    def set_input_message_len(self):
        assert len(self.input_message_len) == 0
        for msg in self.messages:
            self.input_message_len.append(len(msg))
    
    def add_context(self, new_msg, role="user"):
        formatted_msg ={"role": role, "content": new_msg}
        self.messages.append(formatted_msg)

    def chat_with_gpt(self):
        ###################################################
        gpt_response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=self.messages,
            temperature=1,
            max_tokens=MAX_TOKEN,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        ###################################################
        # TODO: Which one is correct?
        # self.response = gpt_response.choices[0].text.strip()
        reason = gpt_response['choices'][0]['finish_reason']
        if reason != "stop":
            print("ERROR: GPT failed, finished_reason: {}")
            print("return None...")
            return None
        self.finished_reason = reason
        self.response = gpt_response['choices'][0]['message']['content']
        self.created_time = gpt_response["created"]
        self.uid = gpt_response["id"]
        self.completion_tokens = gpt_response["usage"]["completion_tokens"]
        self.prompt_tokens = gpt_response["usage"]["prompt_tokens"]
        self.total_tokens = gpt_response["usage"]["total_tokens"]
        print(f"gpt response: {gpt_response}")
        print(f"extracted response: {self.response}")
        return self.response
        
    def write_result(self, output_filepath):
        temp = list()
        assert len(self.messages) == len(self.input_message_len)
        temp.append(f"uid, {self.uid}")
        for i in range(len(self.messages)):
            temp.append(f"message_{i},{self.messages[i]},{self.input_message_len[i]}")
        temp.append(f"input_message_len,{self.input_message_len}")
        temp.append(f"MAX_TOKEN, {MAX_TOKEN}")
        temp.append(f"data, {self.data}")
        temp.append(f"runtime, {self.runtime}")
        temp.append(f"output_text, {self.output_text}")
        temp.append(f"gpt_model, {self.gpt_model}")
        temp.append(f"finished_reason, {self.finished_reason}")
        temp.append(f"response, {self.response}")
        temp.append(f"created_time, {self.created_time}")
        temp.append(f"completion_tokens, {self.completion_tokens}")
        temp.append(f"prompt_tokens, {self.prompt_tokens}")
        temp.append(f"total_tokens, {self.total_tokens}")
        #path_ = util.get_current_time() + "-gpt_output.txt"  # Replace with the path to your file
        path_ = output_filepath
        with open(path_, "w") as file:
            for elem in temp:
                file.write(elem + "\n")
    
    
class GPT:
    def __init__(self):
        self.num_query = 0
        # self.api_endpoint = 'https://api.openai.com/v1/engines/davinci-codex/completions'
    
    def send_request(self, cq, output_filepath):
        '''
        reference: https://platform.openai.com/docs/guides/gpt/chat-completions-api
        The system message helps set the behavior of the assistant. For example, you can modify the personality of the assistant or provide specific instructions about how it should behave throughout the conversation. However note that the system message is optional and the model’s behavior without a system message is likely to be similar to using a generic message such as "You are a helpful assistant."
        '''
        cq.set_input_message_len()
        result = cq.chat_with_gpt()
        #print(result)
        ts = time.time()
        # response = requests.post(self.api_endpoint, json=cq.params, headers=cq.headers)
        # cq.data = response.json() # data is python dictionary. resopnse is json.
        assert cq.runtime == -1
        cq.runtime = (time.time() - ts)
        self.num_query += 1
        cq.write_result(output_filepath)
        return cq.response
    
        
    def call_chatgpt_api(self, query_prompt, output_filepath):
        
        cq = Multi_Message_ChatGPTQuery()
        cq.add_context(query_prompt)
        # cq.add_context(..) # can add more queries
        gpt_output = self.send_request(cq, output_filepath)
        # mongodb_code = gpt_output['choices'][0]['text']
        #print("********************")
        #print("** chatgpt output **")
        #print("********************")
        #print(gpt_output)
    
###****************************************************************************************************
if __name__ == "__main__":

    ################# SETTINGS #############################
    ## I think full paths need to be given # TODO: Fix this 
    CONFIG_FOLDER = "/home/chitty/Desktop/cs598dk/dbknitter/dbknitter/config"
    QUERY_FOLDER    = "/home/chitty/Desktop/cs598dk/dbknitter/dbknitter/query"
    OUTPUT_FOLDER   = "/home/chitty/Desktop/cs598dk/dbknitter/dbknitter/query_output"
    #########################################################
    
    ## Feed in Datalake information (we name it "myData here")
    datalake = Datalake("myData")
    datalake.define_datalake_from_folder(CONFIG_FOLDER)

    ## Create prompt generation object
    prompt = Prompt(datalake)
    gpt = GPT()
    

    query_files=[f for f in os.listdir(QUERY_FOLDER) if os.path.isfile(QUERY_FOLDER+'/'+f)]

    for qfile in query_files:
        query_prompt =prompt.gen_full_prompt(QUERY_FOLDER+'/'+qfile, True)  # To get the query from a file
        #query_prompt = prompt.gen_full_prompt("A_QUERY")      # to just pass query as string
        print("\n\n")
        output_file = OUTPUT_FOLDER+'/'+qfile
        gpt.call_chatgpt_api(query_prompt, output_file)

    # TODO: Fit many queries within same context
    


    
    
    

