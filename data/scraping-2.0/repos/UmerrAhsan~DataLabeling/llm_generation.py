from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
from config import entities
from config import llm_answers_path_n_samples
from config import llm_answers_path


import os
os.environ["OPENAI_API_KEY"] = ""



document_template = '''
Extract the following entities from the given text:
1) {entity1}
2) {entity2}
3) {entity3}
4) {entity4}
5) {entity5} 
6) {entity6}

Instructions:

1)	Output date entity as is if explicitly mentioned in the given text. 

2)	Calculate the Effective date and/or Expiration date from the given text if and only if it can be calculated. Otherwise, output 'NA' for the date.

3)	Only output the state name in case of Governing law entity, Discard any words like 'state' or 'state of’ from the output.

4)	Only output the names of parties not words like licensor, MA etc. in case of Parties entity. Use the # delimiter to separate parties.

5)	If an entity is missing in the given text, Output ‘NA’ for that field.

Strictly follow the above instructions.

Use the following Format:

Example1:

Input text:
['VIDEO-ON-DEMAND CONTENT LICENSE AGREEMENT'] ['EuroMedia Holdings Corp.', 'Rogers', 'Rogers Cable Communications Inc.', 'Licensor'] ['July 11, 2006'] ['July 11, 2006'] ['The term of this Agreement (the Initial Term) shall commence as of the Effective Date and, unless earlier terminated in accordance with this Agreement, shall terminate on June 30, 2010.'] ['This Agreement is subject to all laws, regulations, license conditions and decisions of the Canadian Radio-television and Telecommunications Commission (CRTC) municipal, provincial and federal governments or other authorities which are applicable to Rogers and/or Licensor, and which are now in force or hereafter adopted ('Applicable Law').', 'This Agreement shall be governed by laws of the Province of Ontario and the federal laws of Canada applicable therein.']

Output:
Document name: VIDEO-ON-DEMAND CONTENT LICENSE AGREEMENT
Parties: EuroMedia Holdings Corp. # Rogers Cable Communications Inc. 
Agreement date: 7/11/2006
Effective date: 7/11/2006
Expiration date: 6/30/2010
Governing law: Ontario

Example2:

Input text:
['MARKETING AFFILIATE AGREEMENT'] ['BIRCH FIRST GLOBAL INVESTMENTS INC.', 'MA', 'Marketing Affiliate', 'MOUNT KNOWLEDGE HOLDINGS INC.', 'Company'] ['8th day of May 2014', 'May 8, 2014'] ['This agreement shall begin upon the date of its execution by MA and acceptance in writing by Company'] ['This agreement shall begin upon the date of its execution by MA and acceptance in writing by Company and shall remain in effect until the end of the current calendar year and shall be automatically renewed for successive one (1) year periods unless otherwise terminated according to the cancellation or termination provisions contained in paragraph 18 of this Agreement.'] ['This Agreement is accepted by Company in the State of Nevada and shall be governed by and construed in accordance with the laws thereof, which laws shall prevail in the event of any conflict.'] 

Output:
Document name: MARKETING AFFILIATE AGREEMENT
Parties: BIRCH FIRST GLOBAL INVESTMENTS INC. # MOUNT KNOWLEDGE HOLDINGS INC.
Agreement date: 5/8/2014
Effective date: NA
Expiration date: 12/31/2014
Governing law: Nevada

Be concise.

Input text:
{text}

Output:
'''

llm = ChatOpenAI(model_name='gpt-3.5-turbo-1106', temperature=0.0)
document_prompt = PromptTemplate(template=document_template, input_variables=["text","entity1","entity2","entity3","entity4","entity5","entity6" ])
document_llm_chain = LLMChain(prompt=document_prompt, llm=llm)




def generate_labels_from_llm(data_path):

    df = pd.read_csv(data_path)

    # Initialize an empty list to store dictionaries
    llm_answers_data = []
    count = 0
    for index , row in df.iterrows():
        value = row['text']
        a = document_llm_chain.predict(text=value, entity1 = entities[0], entity2 = entities[1], entity3 = entities[2], entity4 = entities[3], entity5 = entities[4], entity6 = entities[5])
        llm_answers_data.append({'uuid': row['uuid'], 'text': row['text'], 'LLM_Answer': a})
        count = count + 1
        print("Sample Labeled Till yet :", count)

    # Convert the list of dictionaries to a DataFrame
    llm_answers_df = pd.DataFrame(llm_answers_data)
    llm_answers_df.to_csv(llm_answers_path, index=False)



def generate_labels_from_llm_for_n_samples(data_path,n):

    df = pd.read_csv(data_path)

    # Initialize an empty list to store dictionaries
    llm_answers_data = []

    for index, row in df.iterrows():
        if index > n:
            break

        value = row['text']
        a = document_llm_chain.predict(text=value, entity1 = entities[0], entity2 = entities[1], entity3 = entities[2], entity4 = entities[3], entity5 = entities[4], entity6 = entities[5])
        llm_answers_data.append({'uuid': row['uuid'], 'text': row['text'], 'LLM_Answer': a})

    # Convert the list of dictionaries to a DataFrame
    llm_answers_df = pd.DataFrame(llm_answers_data)
    llm_answers_df.to_csv(llm_answers_path_n_samples, index=False)
