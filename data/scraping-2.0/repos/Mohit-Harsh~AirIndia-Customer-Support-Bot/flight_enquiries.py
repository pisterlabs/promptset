import pandas as pd
import numpy as np
import langchain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def test():

    return ("called flight enquiry")


def enquiry(llm, query):

    df = pd.read_json('airindia_flights.json')

    template = """[INST]

    You have to Extract the starting place and destination of the flight from the given question.

    question : {query}

    Use the following format for your response:

    Source : starting city of the flight
    Destination : destination city of the flight

    If you don't know the starting or destination then answer 'Not Provided'.
    dont generate any other text.

    [/INST]
    """

    prompt = PromptTemplate(template=template, input_variables=['query'])
    chain = LLMChain(llm=llm,prompt=prompt)
    res = chain(query)['text'].split('\n')

    source = res[0].split(':')[-1].strip()
    destination = res[1].split(':')[-1].strip()

    if 'not provided' in source.lower():

        source = input('Please enter the source city of your flight : ').strip()

    if 'not provided' in destination.lower():

        destination = input('Please enter the destination of your flight : ').strip() 

    s = df.iloc[np.where(df['source_city']==source)]
    d = s.iloc[np.where(s['destination_city']==destination)]
    data = d.reset_index()

    if len(data) == 0:

        return f"No flights are available from {source} to {destination}"
    
    duration = str(data.iloc[0]['duration'])+' hr'
    dtime = data.iloc[0]['departure_time']
    atime = data.iloc[0]['arrival_time']
    price = "Rs. " + str(data.iloc[0]['price'])
    flight = data.iloc[0]['flight']
    stops = data.iloc[0]['stops']

    context = f"Source City : {source}\nDestination City : {destination}\nDuration of Flight : {duration}\nDeparture Time : {dtime}\nArrival Time : {atime}\nTicket Price : {price}\nStops : {stops}\nFlight : {flight}"

    template2 = """[INST]

    You are AirIndia customer support. Answer the customer's query from the given information.

    information : {context}

    customer : {query}

    [/INST]
    """
    prompt2 = PromptTemplate(template=template2,input_variables=['context','query'])
    chain2 = LLMChain(llm=llm,prompt=prompt2)
    answer = chain2.invoke({'context':context,'query':query})

    return answer