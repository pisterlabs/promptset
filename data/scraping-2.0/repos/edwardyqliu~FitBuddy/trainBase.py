import pandas as pd
import openai
import certifi
import ssl
import asyncio
from aiohttp import TCPConnector, ClientSession
import os
import json
import numpy as np
import graphlib

'''
Assume we have some dataset that we would like to be projected into some n-dimensional basis.  
1) Let GPT rank them s.t. every arbitrary pair of points are compared for each dimension. 
2) Then normalize it the range to [0..1]
3) Then, for every new message generated we use vector embedding similarity to find its position in the rankings. 
'''

GPT_MODEL = "gpt-3.5-turbo"
openai.api_key = os.environ["OPENAI_API_KEY"]

def jsonFunctions(msgs, parameter, temperature = .5):
        sortSchema = [{
            "name":"sort",
            "description": f"Order these messages in terms of {parameter}",
            "parameters":{
                "type":"object",
                "properties":{
                    f"{parameter}":{
                    "type":"string",
                    "description": f"The list of the indicies of the least {parameter} to most {parameter}, delimted by ,",
                    },
                },
            },
            "required":[f"{parameter}"],
            }]
        json_data = {"model": GPT_MODEL, "messages":msgs, "temperature":temperature,"functions":sortSchema,"function_call": {"name":"sort"}}
        return json_data

async def GPT(msgs, parameter):
    json_data = jsonFunctions(msgs, parameter)
    await asyncio.sleep(4)
    result = await openai.ChatCompletion.acreate(**json_data)
    result = result["choices"][0]["message"]
    if "function_call" in result.keys():
        fCall = json.loads((result["function_call"]["arguments"]))
        print(fCall)
        return fCall 
     

async def main():
    sslcontext = ssl.create_default_context(cafile=certifi.where())
    conn = TCPConnector(ssl_context=sslcontext)
    session = ClientSession(connector=conn)
    openai.aiosession.set(session)
    #import data
    df = pd.read_csv("training_data/message1.csv")
    #scramble the dataframe
    tasks = []
    parameters = ["aggressiveness","encouraging","deceptive","quirky"]
    for par in parameters:
        RUNS = 100
        for i in range(RUNS):
            N = len(df)
            M = 1
            df_temp = df.sample(frac = M)
            n = 10
            list_df = [df_temp[k:k+n] for k in range(0,len(df_temp),n)]
            for df_i in list_df:
                json_l = [{"role":"system","content":f"Please order these messages in order of {par}"}]
                for j,d in df_i.iterrows():
                    json_l.append({"role":"system","content":f"index {j,d['Message']}"})
            await asyncio.sleep(.1)
            tasks.append(asyncio.create_task(GPT(json_l,par)))

    result = await asyncio.gather(*tasks)

    #We use Erdos Connectivity ~  Estimate for how much of the sample space we've explored
    sum_edges_aggressive = 0
    sum_edges_encouraging = 0
    sum_edges_deceptive = 0
    sum_edges_quirky = 0


    aggr_list = {}
    encouraging_list = {}
    deceptive_list = {}
    quirky_list = {}


    goal = np.log(N)/N

    for row in result:
            for par in parameters:
                 if list(row.keys())[0] == "aggressiveness":
                    a = str(*list(row.values())).split(",")
                    for val in a:
                        if val not in aggr_list:
                            aggr_list[val] = 0
                        aggr_list[val] += a.index(val)
                    sum_edges_aggressive += (len(*row.values())*len(*row.values()) - 1) / 2
                 elif list(row.keys())[0] == "encouraging":
                    e = str(*list(row.values())).split(",")
                    for val in e:
                        if val not in encouraging_list:
                            encouraging_list[val] = 0
                        encouraging_list[val] += e.index(val)
                    sum_edges_encouraging += (len(*row.values())*len(*row.values()) - 1) / 2
                 elif list(row.keys())[0] == "deceptive":
                    d = str(*list(row.values())).split(",")
                    for val in d:
                        deceptive_list[val] = 0
                        if val not in deceptive_list:
                            deceptive_list[val] = 0
                        deceptive_list[val] += d.index(val)
                    sum_edges_deceptive += (len(*row.values())*len(*row.values()) - 1) / 2
                 else:
                    q = str(*list(row.values())).split(",")
                    for val in q:
                        quirky_list[val] = 0
                        if val not in quirky_list:
                            quirky_list[val] = 0
                        quirky_list[val] += q.index(val)
                    sum_edges_quirky += (len(*row.values())*len(*row.values()) - 1) / 2

    diff_aggressive = np.abs(goal - sum_edges_aggressive/((N*(N-1))/2))/goal
    diff_encouraing = np.abs(goal - sum_edges_encouraging/((N*(N-1))/2))/goal
    diff_deceptive = np.abs(goal - sum_edges_deceptive/((N*(N-1))/2))/goal
    diff_quirky = np.abs(goal - sum_edges_quirky/((N*(N-1))/2))/goal

    print("About",diff_aggressive,"% off a connected graph for aggressive")      
    print("About",diff_encouraing,"% off a connected graph for encouraging")      
    print("About",diff_deceptive,"% off a connected graph for deceptive")   
    print("About",diff_quirky,"% off a connected graph for quirky")  

    print()

    aggr_pd = pd.DataFrame(aggr_list.items(),columns = ["index","aggr"]).astype(int)
    aggr_pd.set_index("index",inplace=True)
    encouraging_pd = pd.DataFrame(encouraging_list.items(), columns = ["index","encour"]).astype(int)
    encouraging_pd.set_index("index",inplace=True)
    deceptive_pd = pd.DataFrame(deceptive_list.items(), columns = ["index","decept"]).astype(int)
    deceptive_pd.set_index("index",inplace=True)
    quirky_pd = pd.DataFrame(quirky_list.items(), columns = ["index","quirky"]).astype(int)
    quirky_pd.set_index("index",inplace=True)
    aggr_pd = (aggr_pd - aggr_pd.min()) / (aggr_pd.max() - aggr_pd.min())
    encouraging_pd = (encouraging_pd - encouraging_pd.min()) / (encouraging_pd.max() - encouraging_pd.min())
    deceptive_pd = (deceptive_pd - deceptive_pd.min()) / (deceptive_pd.max() - deceptive_pd.min())
    quirky_pd = (quirky_pd - quirky_pd.min()) / (quirky_pd.max() - quirky_pd.min())
    print(aggr_pd.sort_index(),encouraging_pd.sort_index(),deceptive_pd.sort_index(),quirky_pd.sort_index())

    df = df.set_index("index").combine_first(aggr_pd).reset_index()
    df = df.set_index("index").combine_first(encouraging_pd).reset_index()
    df = df.set_index("index").combine_first(deceptive_pd).reset_index()
    df = df.set_index("index").combine_first(quirky_pd).reset_index()
    print(df)
    df.fillna(.5,inplace=True)
    df.to_csv("training_data/message1Results.csv")
    await openai.aiosession.get().close()
    return df
    
asyncio.run(main())