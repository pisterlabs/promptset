import requests
import urllib3
import stapp #file
from dotenv import dotenv_values
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) #I forget why I added this
config = dotenv_values(".env") #pip install chromadb, tabulate google-search-results reportlab openai

import pandas as pd #dataframe
import numpy as np
import re
import datetime as dt #datetime for formatting iso 8601 date
from datetime import date #convert seconds to mins, hours, etc

import streamlit as st
from langchain import  LLMChain #SerpAPIWrapper,
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, create_pandas_dataframe_agent
#from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
#from langchain.document_loaders import TextLoader
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI #, GPT4All
from langchain.prompts import BaseChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage

from typing import List, Union
import os
import base64
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (Mail, Attachment, FileContent, FileName, FileType, Disposition)
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter

activities_url = "https://www.strava.com/api/v3/athlete/activities"


def convert_to_miles(num):
    return ((num/m_conv_factor)*1000)/1000.0
def todays_date():
    todaysDate = dt.datetime.now().strftime('%Y, %m, %d')
    return todaysDate

def calc_days_till_marathon(): 
    todayDate = todays_date()
    date_object = dt.datetime.strptime(todayDate, '%Y, %m, %d').date()
    d1 = date(2023, 12, 10)
    delta = d1 - date_object
    return delta

def calc_weeks_till_marathon():
    daysToM = calc_days_till_marathon()
    return daysToM/7
#number of workouts in Strava data
def num_rows_in_dataframe(df):
    return len(df.index)
email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
def validate_email(email):
    # pass the regular expression and the string into the fullmatch() method
    if(re.fullmatch(email_regex, email)):
        return True
    else:
        return False

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Marathon Day" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Marathon Day:")[-1].strip()},
                log=llm_output,
            )
        elif marathon_date in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split(marathon_date)[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            try:
                action = match.group(1).strip()
                action_input = match.group(2)
                # Return the action and action input
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
            except ValueError as e:
                response = str(e)
                if not response.startswith("Could not parse LLM output: `"):
                    raise e
                response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
       

tools = [
      Tool(
        name = "get today's date",
        func = lambda x:todays_date(),
        description="use to get today's date"
    ),
    Tool(
        name = "days to marathon",
        func = lambda x: calc_days_till_marathon(),
        description="use to calculate the number of days from today until the marathon"
    ),
    Tool(
        name = "weeks to marathon",
        func = lambda x:calc_weeks_till_marathon(),
        description="use to get the number of weeks from today until the marathon"
    ),
    Tool(
        name = "rows in csv",
        func = lambda df: num_rows_in_dataframe(df),
        description="use to get the number of rows in csv file to calculate averages from running data"
    ) #,
    #search_tool
]

st.title('Personal Marathon Training plan generator')
st.subheader('enter details below')

with st.form('my_form'):
    strava_token_input = st.text_input('Strava API token')
    email = st.text_input('Email to send plan to')
    marathon_date = st.text_input('When is your marathon date?')
    training_start_date = st.text_input('When do you want to start training?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        if not validate_email(email):
            st.error("invalid email", icon="🚨")
            st.cache_data()
        elif training_start_date > marathon_date:
            st.error("Start date must be earlier than end date", icon="🚨")
            st.cache_data()
        else:
            success = st.success("Thank you for inputting valid dates and emails! Plan is being generated.") 

            header = {'Authorization': 'Bearer ' + strava_token_input} #config.get('STRAVA_TOKEN')}
            params = {'per_page': 200, 'page': 1} #max 200 per page, can only do 1 page at a time
            my_dataset = requests.get(activities_url, headers=header, params=params).json() #activities 1st page
            page = 0
            for x in range(1,5): #loop through 4 pages of strava activities
                page +=1 
            params = {'per_page': 200, 'page': page}
            my_dataset += requests.get(activities_url, headers=header, params=params).json() #add to dataset, need strava token in .env to be updated else get dict error
    
            activities = pd.json_normalize(my_dataset)
            # print(activities.columns) # list all columns in the table
            # print(activities.shape) #dimensions of the table.

            #Create new dataframe with specific columns #max_time
            cols = ['name', 'type', 'distance', 'moving_time', 'total_elevation_gain', 'start_date']
            activities = activities[cols]
            activities = activities[activities["start_date"].str.contains("2021") == False] #remove items from 2021, only include workouts from 2022 and 2023
            activities.to_csv('data_files/activities.csv', index=False)

            # loop through activities data frame to get number of activities of each type
            num_runs = len(activities.loc[activities['type'] == 'Run'])
            num_walks = len(activities.loc[(activities['type'] == 'Walk') & (activities['total_elevation_gain'] > 90)])
            num_rides = len(activities.loc[activities['type'] == 'Ride'])
            num_elliptical = len(activities.loc[activities['type'] == 'Elliptical'])
            num_weight_training = len(activities.loc[activities['type'] == 'WeightTraining'])
            num_swims = 0
            num_tennis = 0
            for i in activities['name'].values:
                if 'swim' in i.lower():
                    num_swims +=1
                if 'tennis' in i.lower():
                    num_tennis +=1
            cross_training_options = activities['type'].unique()
            # make CSV of runs
            runs = activities.loc[activities['type'] == 'Run']
            runs.to_csv('data_files/runs.csv', index=False) #index=False writes out weird unnamed index column in pandas df

            #convert meters to miles
            data_df = pd.read_csv('data_files/runs.csv')
            m_conv_factor = 1609

            data_df['distance'] = data_df['distance'].map(lambda x: convert_to_miles(x))
            #convert moving time secs to mins, hours
            #data_df['moving_time'] = data_df['moving_time'].astype(str).map(lambda x: x[7:]) #slices off 0 days from moving_time
            data_df.to_csv('data_files/runs.csv')

            os.environ["OPENAI_API_KEY"] = config.get('OPENAI_API_KEY')
            # number of days for workouts
            avg_distance = data_df['distance'].mean()
            avg_moving_time = data_df['moving_time'].mean()
            max_distance_ran = data_df['distance'].max()
            avg_miles = []
            for a,b in zip(data_df.distance, data_df.moving_time):
                avg_miles.append((b/a)/60)
                avg_mile= sum(avg_miles) / len(avg_miles)
            print('avg_distance ', avg_distance)
            print('avg_moving_time ', avg_moving_time)
            print('avg_mile', avg_mile)

            llm = OpenAI(temperature=0)
            pd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), data_df, verbose=True) #csv agent?
            # csv agent can be used to load data from CSV files and perform queries, while the Pandas Agent can be used to load data from Pandas data frames and process user queries. Agents can be chained together to build more complex applications.

            pd_output = pd_agent.run("Calculate avg distance, avg moving time, max moving time, max distance")
        
            coach_template = """
            You are a marathon trainer. 
            Use {pd_output} to customize a marathon training plan.
            Weekly mileage should eventually be 45 miles a week. Here is context about your runner:
            User: training start date: {training_start_date}, marathon date: {marathon_date}
            AI: {plan}
            """

            example_prompt = PromptTemplate(input_variables=["marathon_date", "training_start_date", "pd_output", "plan"], template=coach_template)

            prefix = """
            The following are example marathon training plans based on Strava data with an AI
            assistant. The plans slowly increase mileage each week, slowly increase the weekly long run, and include a weekly long run. Some days have runs, other days have cross-training.
            The longest long run is around 18 miles.
            Here are some examples: 
            """
            suffix = """
            User: {training_start_date}
            User: {marathon_date}
            AI: """
            few_shot_prompt_template = FewShotPromptTemplate(
                examples=stapp.examples, 
                example_prompt=example_prompt, 
                prefix=prefix,
                suffix=suffix, 
                input_variables=["marathon_date", "training_start_date"]
            )
            output_parser = CustomOutputParser()
            # LLM chain consisting of the LLM and a prompt
            llm_chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)
            tool_names = [tool.name for tool in tools]
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain, 
                output_parser=output_parser,
                stop=["\Final Answer:"], 
                allowed_tools=tool_names
            )
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
            plan = agent_executor({"marathon_date":marathon_date, "training_start_date": training_start_date, "pd_output": pd_output})
            plan = plan['output']
            plan = str(plan).replace('\n','<br />') #each workout on new line

            message = Mail(
                from_email='langchain_sendgrid_marathon_trainer@sf.com',
                to_emails=email,
                subject='Your AI-generated marathon training plan',
                html_content='<strong>Good luck at your marathon on %s</strong>!\n\nYour plan is attached.'%(marathon_date))

            styles = getSampleStyleSheet()
            styleN = styles['Normal']
            styleH = styles['Heading1']
            story = []

            pdf_name = 'plan.pdf'
            doc = SimpleDocTemplate(
                pdf_name,
                pagesize=letter,
                bottomMargin=.4 * inch,
                topMargin=.6 * inch,
                rightMargin=.8 * inch,
                leftMargin=.8 * inch)
            P = Paragraph(plan['output'], styleN)
            story.append(P)

            doc.build(
                story,
            )
            with open(pdf_name, 'rb') as f:
                data = f.read()
                f.close()
            encoded_file = base64.b64encode(data).decode()

            attachedFile = Attachment(
                FileContent(encoded_file),
                FileName('attachment.pdf'),
                FileType('application/pdf'),
                Disposition('attachment')
            )
            message.attachment = attachedFile
            os.environ["SENDGRID_API_KEY"] = config.get('SENDGRID_API_KEY')
            sg = SendGridAPIClient()
            response = sg.send(message)
            code, body, headers = response.status_code, response.body, response.headers
            print(f"Response Code: {code} ")
            print(f"Response Body: {body} ")
            print(f"Response Headers: {headers} ")
            print("Message Sent!")