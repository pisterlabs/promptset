==> db/db_manager.py <==
import pandas as pd
from sqlalchemy import create_engine
import glob

class XLSProcessor:
    def __init__(self, db_manager):
        self.db_manager = db_manager  # Assume db_manager is an instance of DBManager

    def read_and_load(self, directory_path, table_name):
        # Using glob to get all .xls files in the specified directory
        file_paths = glob.glob(f"{directory_path}/*.xls")
        for file_path in file_paths:
            print('running on ', file_path)
            # Reading xls file
            # data_frame = pd.read_excel(file_path, engine='xlrd')
            data_frame = pd.read_html(file_path)[0]
            fp_csv = '..' + file_path.split('.')[2] + '.csv' ## NOTE! Only works on subfolder, skipping ../
            data_frame.to_csv(fp_csv, index=False)

            # You might want to process or clean your data here
            
            # Loading data to psql
            self.db_manager.insert_data(data_frame, table_name)

class DBManager:
    def __init__(self, dbname, user, password, host):
        self.engine = create_engine(f'postgresql://{user}:{password}@{host}/{dbname}')

    def insert_data(self, data_frame, table_name='your_table_name'):
        data_frame.to_sql(table_name, self.engine, if_exists='append', index=False)  # Assume no index for simplicity

# Usage:
db_manager = DBManager(dbname='accsf_user', user='accsf_user', password='acceleranto2!', host='localhost')
xls_processor = XLSProcessor(db_manager)
xls_processor.read_and_load('../data', 'legistar_scraper')  # Provide your directory path


==> db/db_manager_supabase.py <==
from supabase import create_client
import pandas as pd

import os

class SupabaseConnector():
    def __init__(self, url, key):
        self.supabase = create_client(url, key)
    
    def read_table(self, table_name, query='*'):
        data, count = self.supabase.table(table_name).select(query).execute()
        return pd.DataFrame(data[1])

    def write_rows(self, data_frame, table_name):
        self.supabase.table(table_name).insert(data_frame).execute()


if __name__=='__main__':
    supa = SupabaseConnector(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

    df = supa.read_table('aaron_peskin')


==> processor/llm_processor.py <==
import os
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.chains import LLMChain

import pandas as pd

class LLMProcessor:
    def __init__(self, data):
        self.data = pd.read_csv(data, nrows=5) ## Testing on 5 rows

    def initialize_agent(self, agent_def_path):
        """Initialize the agent and return it."""
        ## Load prompts
        with open(agent_def_path) as f:
            system_prompt = f.read()
        
        # Setup the chat prompt with memory
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),  # The persistent system prompt
            MessagesPlaceholder(variable_name="chat_history"),  # Where the memory will be stored
            HumanMessagePromptTemplate.from_template("{human_input}"),  # Human input
        ])

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        llm = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            temperature=0,
            request_timeout=120,
        )
        chat_llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )
        self.agent = chat_llm_chain
    
    def run_agent(self, agent, human_input):
        """Run the provided agent using the input and return the result."""
        result = agent.predict(human_input=human_input)
        return result

    def apply_run_agent(self, row):
        return self.run_agent(self.agent, row['Title'])

    def process(self):
        # Run LLM on combined fields and create summary table
        # This is a placeholder, update with actual processing logic

        ## Change to apply
        # llm_response = self.run_agent(self.agent, self.data['Title'][1])
        self.data['llm_response'] = self.data.apply(self.apply_run_agent, axis=1)
        # print(llm_response)


        return {"summary": {}}


test_processor = LLMProcessor('../data/csv/Ahsha_Safai.csv')
test_processor.initialize_agent('./agent_prompts/summarizer.txt')

test_processor.process()

## 




==> processor/outcome_extractor.py <==
class OutcomeExtractor:
    def __init__(self, data):
        self.data = data

    def extract(self):
        # Extract concrete outcomes (investment, impact, etc)
        # This is a placeholder, update with actual extraction logic
        return {"outcomes": {}}

==> processor/reliability_rating.py <==
class ReliabilityRating:
    def __init__(self, data):
        self.data = data

    def calculate(self):
        # Assign/calc reliability rating
        # This is a placeholder, update with actual calculation logic
        return {"reliability_rating": 0}

==> processor/vertical_labeler.py <==
class VerticalLabeler:
    def __init__(self, data):
        self.data = data

    def label(self):
        # Add vertical label
        # This is a placeholder, update with actual labeling logic
        return {"vertical_label": ""}

==> scraper/data_api_scraper.py <==
import requests

class DataAPIScraper:
    def __init__(self, url):
        self.url = url

    def scrape(self):
        response = requests.get(self.url)
        # Scrape data from API
        # This is a placeholder, update with actual scraping logic
        return {"data": []}

==> scraper/legistar_scraper.py <==
import requests
from bs4 import BeautifulSoup

class MyWebScraper:
    def __init__(self, url):
        self.url = url

    def scrape(self):
        response = requests.get(self.url)
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve page with status code: {response.status_code}")

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the element with the desired ID
        starting_element = soup.find(id='ctl00$ContentPlaceHolder1$gridDepartments$ctl00$ctl02$ctl01$ctl00')
        breakpoint()

        # Navigate to its parent element
        tbody = starting_element

        # Loop over all tr elements that are children of tbody
        for row in tbody.find_all('tr', recursive=False):
            columns = row.find_all('td')
            field_texts = [col.text.strip() for col in columns]
            print(' | '.join(field_texts))

# Usage:
url = "https://sfgov.legistar.com/PersonDetail.aspx?ID=196476&GUID=63B530EB-D641-42BB-BFEB-A1367DC844CE&Search="
scraper = MyWebScraper(url)
scraper.scrape()

 
==> scraper/resolutions_scraper.py <==
import requests
from bs4 import BeautifulSoup

class ResolutionsScraper:
    def __init__(self, url):
        self.url = url

    def scrape(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Scrape resolutions
        # This is a placeholder, update with actual scraping logic
        return {"resolutions": []}

==> scripts/fetch_data.py <==
import os
from supabase import create_client, Client

url: str = "https://hxrggsnimtifedjvpupp.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh4cmdnc25pbXRpZmVkanZwdXBwIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTkxMzU5NDYsImV4cCI6MjAxNDcxMTk0Nn0.eAZYkghBmpIVkED5QWvjcMyFvsfcpgiyFSNADV3AGEA"
supabase: Client = create_client(url, key)

response = supabase.table('aaron_peskin').select("*").execute()
print(response.data)