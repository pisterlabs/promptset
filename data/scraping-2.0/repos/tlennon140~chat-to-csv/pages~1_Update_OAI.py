# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit.logger import get_logger
from pandasai import SmartDataframe
from pandasai.llm import AzureOpenAI
from openai import AzureOpenAI as OpenAIAzure


def oai() -> None:

    load_dotenv()  # This loads the .env file at the project root

    expected_headers = [
        "Issue Name", "Issue Number", "Issue Criteria", "Issue Condition", 
        "Issue Cause", "Issue Effect", "Priority", "Recommendation", 
        "Management Action Plan", "Estimated Completion Date"
    ]

    client = OpenAIAzure(
        api_key=os.environ.get('AZURE_UNDP_API_KEY'), 
        api_version="2023-07-01-preview",
        azure_endpoint = os.environ.get('AZURE_ENDPOINT')
    )
        
    llm = AzureOpenAI(
        api_token=os.environ.get('AZURE_UNDP_API_KEY'),
        azure_endpoint=os.environ.get('AZURE_ENDPOINT'),
        api_version="2023-05-15",
        deployment_name=os.environ.get('AZURE_UNDP_MODEL')
    )
    image_path = "./exports/charts/temp_chart.png"

    system_prompt = '''
    You are UNDP Audit Data Extraction Tool. Your job is to extract the recommendation titles from the text. Make sure you extract the titles and not the descriptions. It is extremely important you return the precise response format below without any further information or analysis because this structured response is fed into other tools that expect this precise response format:

    Recommendation 1: {Recommendation 1 title} 
    Recommendation 2: {Recommendation 2 title} 
    Recommendation 3: {Recommendation 3 title} 
    ... Continue until all recommendations are listed.

    Audit report:
    '''

    def extract_recommendation_titles(audit_report, model):
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'assistant', 'content': audit_report},
        ]

        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4-cdo",
            max_tokens=4000
        )


        response_text = response.choices[0].message.content
        pattern = r'Recommendation \d+: (.+?)(?=\s*Recommendation \d+:|$)'
        titles = re.findall(pattern, response_text)
        return titles

    def get_recommendation_details(title, audit_report, file_name, model):
        detailed_prompt = f'''
        You are a UNDP Audit Report analysis expert. Review this audit report and for the recommendation "{title}" provide the following information, but do not summarize or use your own words, but take them directly from the audit report that has been provided for context. Give it in this format, and do not provide any introductory remarks before providing the format.

        Issue Name: {{IssueName}}
        Issue Number: Issue number as a digit without any words.
        Issue Criteria: {{The UNDP regulation or requirements that is related to the issue}}
        Issue Condition: What was found in the audit that broke the UNDP regulation?
        Issue Cause: What caused the condition?
        Issue Effect: What is the impact / risk if this condition continues as it is
        Priority: {{Priority Level}}
        Recommendation: {{Recommendation Text}}
        Management Action Plan: {{Management Action Plan}}
        Estimated Completion Date: {{Estimated Completion Date}}
        '''

        messages = [
            {'role': 'system', 'content': detailed_prompt},
            {'role': 'assistant', 'content': audit_report},
        ]
        
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4-cdo",
            max_tokens=4000
        )
    
        response_text = response.choices[0].message.content
        return response_text

    def parse_record(record):
        lines = record.strip().split('\n')
        record_dict = {}
        for line in lines:
            if ':' in line and not line.startswith("Recommendation "):  # Exclude "Recommendation N" lines
                key, value = line.split(':', 1)
                record_dict[key.strip()] = value.strip()
        return record_dict       

    def add_to_df(df, processed_data, file_path):

        records = processed_data.split('--------------------------------------------------')
        parsed_records = [parse_record(record) for record in records if record.strip()]
        temp_df = pd.DataFrame(parsed_records)
        actual_headers = temp_df.columns.tolist()
        #temp_df['Audit ID'] = file_name.rstrip(".txt")
                    
        if actual_headers != expected_headers:
            print("Incorrect headers")
            return df     
        else:
            df = pd.concat([df, temp_df], axis=0)
            return df



    st.sidebar.success("Select a dataset above.")

    st.markdown(
        """
        Welcome to the Update OAI Report Page

        Please upload the doc that you would like added to the dataset on file.
    """
    )

    #st.image(image_path)

    df = pd.read_csv("./data/audits.csv")
    st.write(df.head(3))   
    
    uploaded_file = st.file_uploader("Upload a new audit report PDF", type=['pdf'])
    if uploaded_file is not None:
      df = pd.read_csv(uploaded_file)
      st.write(df.head(3))
      prompt = st.text_area("Please enter your question:")

      smart_df = SmartDataframe(df, config={"llm": llm})

      # Generate output
      
      if st.button("Generate"):
          if prompt:
              with st.spinner("Generating response..."):
                  code = smart_df.chat(prompt)
                  #st.write(pandas_ai.run(df, prompt))
                  st.write(code)

                  if os.path.exists(image_path):
                    # Display the image
                    st.image(image_path)
                  
          else:
              st.warning("Please enter a prompt.")




st.set_page_config(page_title="Update OAI", page_icon="📹")
st.markdown("# Update OAI CSV")
st.sidebar.header("Update OAI")

oai()
