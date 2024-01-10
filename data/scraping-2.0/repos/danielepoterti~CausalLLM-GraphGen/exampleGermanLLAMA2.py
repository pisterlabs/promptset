from core.discovery import get_graphs
from langchain.llms import Replicate
import networkx as nx
import pandas as pd
import os


def load_germandataset(nodes):
    '''
    read dataset and preprocessing for german credit dataset
    return data only for the nodes
    '''
    
    df = pd.read_csv("data/german_data_credit_dataset.csv")
    #create quickaccess list with categorical variables labels
    catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
               'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job',
               'telephone', 'foreignworker']
    #create quickaccess list with numerical variables labels
    numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age',
               'existingcredits', 'peopleliable', 'classification']

    # Binarize the target 0 = 'bad' credit; 1 = 'good' credit
    df.classification.replace([1,2], [1,0], inplace=True)


    #  dic categories Index(['A11', 'A12', 'A13', 'A14'], dtype='object')
    dict_categorical = {}
    for c in catvars:
        dict_categorical[c] = list(df[c].astype("category").cat.categories)
        df[c] = df[c].astype("category").cat.codes

    #  create gender variable 1= female 0 = male

    df.loc[df["statussex"] == 0, "gender"] = 0
    df.loc[df["statussex"] == 1, "gender"] = 1
    df.loc[df["statussex"] == 2, "gender"] = 0
    df.loc[df["statussex"] == 3, "gender"] = 0
    df.loc[df["statussex"] == 4, "gender"] = 1

    #  all features as float
    df = df.astype("float64")
    df["classification"] = df["classification"].astype("int32")
    # save codes
    with open('dict_german.txt', 'w') as f:
        f.write(str(dict_categorical))

    return df[nodes]

descriptions = {
    "gender": "Individual's gender, potentially influencing risk profile, financial inclusion, and product design requirements in credit lending.",

    "age": "Individual's age, affecting eligibility, health status, and financial behavior which could impact credit risk assessment.",

    "creditamount": "Requested loan amount, providing insight into the individual's borrowing needs and repayment capacity.",

    "duration": "Loan repayment period, reflecting the loan term from disbursement to final payment, crucial in assessing repayment capability.",

    "classification": "Derived prediction of creditworthiness based on other features, used to gauge the individual's risk level for lenders and estimate probability of timely repayment or default. This feature cannot cause any variation in other features.",
}


df = load_germandataset(["gender", "age", "creditamount", "duration", "classification"])

immutable_features = ["gender", "age"]



llm = Replicate(
    model="replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
    input={"temperature": 0.01, "system_prompt": ""},
)

result = get_graphs(df, descriptions, immutable_features, "credit lending in germany", "classification", "results", llm)



