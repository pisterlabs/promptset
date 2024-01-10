from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
from openai import OpenAI
os.environ['OPENAI_API_KEY'] = ''

import pandas as pd
import numpy as np




function_json = {
  "name": "get_lowest_offer",
  "parameters": {
    "type": "object",
    "properties": {
      "final_cost": {
        "type": "number",
        "description": "What is the best offer/lowest cost the vendor can offer."
      },
      "final_days": {
        "type": "number",
        "description": "What is the best offer/minimum number of days the vendor can deliver in."
      }
    },
    "required": [
      "final_cost",
      "final_days"
    ]
  },
  "description": "Collect the cost and delivery days from the vendor after the negotiations."
}

def initiate_client(their_offer, our_offer):
    client = OpenAI(api_key="")
    for i in [i.id for i in client.beta.assistants.list().data if i.name == "Summarization_Assistant_ani"]:
        client.beta.assistants.delete(i)
    assistant = client.beta.assistants.create(
        name="Summarization_Assistant_ani",
        instructions=f"You are an AI assistant who is supposed to negotiate with the vendor. The vendors best offer is {their_offer}. We want to negotiate it down to {our_offer}. You are supposed to collect the counter offer from the vendor: Can the lowest cost or time be met, if not, what is the lowest they can offer. Do not write them a new counter offer. Collect the information and invoke the function. Always write your responses in the form of a mail on behalf of Vishwa Singh.",
        model="gpt-4-1106-preview",
        tools=[
            {"type": "function", "function": function_json},
        ],
    )

    MATH_ASSISTANT_ID = assistant.id  
    thread = client.beta.threads.create()
    
    return client, MATH_ASSISTANT_ID, thread

def displayMinVendor(ven):
    ven["Unfulfilled_len"] = ven['Unfulfilled'].apply(lambda x: len(str(x).split(';')) if pd.notna(x) else 0)

    ven['Cost' + '_normalized'] = normalize_by_sum(ven['Cost'])
    ven['Days' + '_normalized'] = normalize_by_sum(ven['Days'])
    ven['Unfulfilled_len' + '_normalized'] = normalize_by_sum(ven['Unfulfilled_len'])

    
    ven['Overall'] = ven.apply(lambda row: 0.4 * row['Cost_normalized'] +
                                       0.35 * row['Days_normalized'] +
                                       0.25 * row['Unfulfilled_len_normalized'], axis=1)
    
    
    matching_row = ven[(ven['Cost'] == ven['Cost'].min()) & (ven['Days'] == ven['Days'].min())]

    if not matching_row.empty:
    # If a matching row exists, print the row
        print("Matching Row:")
        print(matching_row)

        return matching_row[['VendorID', 'Cost', 'Days', 'Unfulfilled']], (matching_row['Cost'].min(), matching_row['Days'].min())
    else:
    # If no matching row exists, print the minimum values
        min_values_Cost = ven['Cost'].min()
        min_values_Days = ven['Days'].min()
        print(f"Minimum Cost Offered: {min_values_Cost}, Mimimum Days Offered: {min_values_Days}")
    
        min_values = ven['Overall'].nsmallest(1)
        result_rows = ven[ven['Overall'].isin(min_values)]

        return result_rows[['VendorID', 'Cost', 'Days', 'Unfulfilled']], (min_values_Cost, min_values_Days)
    
def normalize_by_sum(column):
    normalized_column = column / column.sum()
    return normalized_column



def get_counter_offer(offer):
    ven = pd.read_csv("C:/VS code projects/Road to Hack/auto_negotiator/Utilities/Vendor_yes.csv")
    ## Check if the requirements are satisfied
    if offer["requirements_satisfied"]:
        ## Convert offer to dataframe
        dataframe = pd.DataFrame(np.array([['NEW'] + list(offer.values())])[:,[0,-2,-1,1,2]], columns=['VendorID', 'Cost', 'Days', 'CanFulfill', 'Unfulfilled'],index=[len(ven)])
        #Replace True with T and False with F
        dataframe['CanFulfill'] = dataframe['CanFulfill'].astype(str).str[0].str.upper()
        
        ## Append the offer to the vendor dataframe
        ven = ven.append(dataframe, ignore_index=True)

        ## Convert cost and days to numeric
        ven['Cost'] = pd.to_numeric(ven['Cost'])
        ven['Days'] = pd.to_numeric(ven['Days'])

        ## Get minimum vendor
        min_vendor, min_values = displayMinVendor(ven)

        if min_vendor['VendorID'].iloc[0] == 'NEW':
            ## Check if min_vendor is better than min_values
            if min_vendor['Cost'].iloc[0] > min_values[0] or min_vendor['Days'].iloc[0] > min_values[1]:
                ## Create a new offer in {'new_cost':3,'new_days':14} format
                return True, {'new_cost':min(min_vendor['Cost'].iloc[0], min_values[0]),'new_days':min(min_vendor['Days'].iloc[0], min_values[1])}, {'previous_cost':min_vendor['Cost'].iloc[0],'previous_days':min_vendor['Days'].iloc[0]}
            else:
                return False, {}
        else:
            False, {}
        


def gpt_negotiation_mail(their_offer, our_offer, vendor_name):
    # Create a GPT prompt
    prompt = f"Write a mail to a vendor named '{vendor_name}' on behalf of Vishwa Mohan Singh (salutations), asking them to negotiate from:/nprevious cost: {their_offer['previous_cost']} Euros/n and previous days required {their_offer['previous_days']} to new offer:from:/nprevious cost: {our_offer['new_cost']} Euros/n and previous days required {our_offer['new_days']}"

    mail_assistant = ChatOpenAI()
    messages = [
        SystemMessage(
            content="You are an AI assistant that is supposed to write a mail to the vendor negotiating for a reduced cost and reduced time of delivery."
        ),
        HumanMessage(content=prompt),
    ]
    response = mail_assistant(messages)

    return response.content