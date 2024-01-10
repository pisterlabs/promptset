import networkx as nx
import pandas as pd
import numpy as np
import os
from core.discovery import get_graphs
from langchain.chat_models import ChatOpenAI

def graph_adult_income():
    '''
    returns a graph for the adult income dataset. causal graph obtained from https://arxiv.org/pdf/1611.07438.pdf
    '''

    G = nx.DiGraph(directed=True)

    G.add_node("ethnicity")
    G.add_edges_from([("ethnicity", "income"), ("ethnicity", "occupation"), ("ethnicity", "marital-status"), ("ethnicity", "hours-per-week"), ("ethnicity", "education")])

    G.add_node("age")
    G.add_edges_from([("age", "income"), ("age", "occupation"), ("age", "marital-status"), ("age", "workclass"), ("age", "education"),
                      ("age", "hours-per-week"), ("age", "relationship")])

    G.add_node("native-country")
    G.add_edges_from([("native-country", "education"), ("native-country", "workclass"), ("native-country",  "hours-per-week"),
                      ("native-country", "marital-status"), ("native-country", "relationship"), ("native-country", "income") ])

    G.add_node("gender")
    G.add_edges_from([("gender", "education"), ("gender", "hours-per-week"), ("gender", "marital-status"), ("gender", "occupation"),
                      ("gender", "relationship"), ("gender", "income") ])

    G.add_node("education")
    G.add_edges_from([("education", "occupation"), ("education", "workclass"), ("education", "hours-per-week" ), ("education", "relationship"),
                      ("education", "income") ])

    G.add_node("hours-per-week")
    G.add_edges_from([("hours-per-week", "workclass"), ("hours-per-week", "marital-status" ), ("hours-per-week", "income")])

    G.add_node("workclass")
    G.add_edges_from([("workclass", "occupation"), ("workclass", "marital-status" ), ("workclass", "income")])

    G.add_node("marital-status")
    G.add_edges_from([("marital-status", "occupation"), ("marital-status", "relationship"), ("marital-status", "income")])

    G.add_node("occupation")
    G.add_edges_from([("occupation", "income")])

    G.add_node("relationship")
    G.add_edges_from([("relationship", "income")])

    G.add_node("income")
    return G


def load_adultdataset(nodes):
    '''
    read dataset and preprocessing for adult income dataset
    return data only for the nodes
    '''

    df = pd.read_csv("data/adult_income_dataset.csv")
    print(df.shape)
    # Binarize the target 0 = <= credit; 1 = >50K
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
    # Finding the special characters in the data frame
    df.isin(['?']).sum(axis=0)

    # code will replace the special character to nan and then drop the columns
    df['native-country'] = df['native-country'].replace('?', np.nan)
    df['workclass'] = df['workclass'].replace('?', np.nan)
    df['occupation'] = df['occupation'].replace('?', np.nan)
    # dropping the NaN rows now
    df.dropna(how='any', inplace=True)
    print(df.shape)


    # categorical variables
    catvars = ['workclass',  'marital-status', 'occupation', 'relationship',
               'ethnicity', 'gender', 'native-country']
    # education order > https: // www.rdocumentation.org / packages / arules / versions / 1.6 - 6 / topics / Adult
    df['education'] = df['education'].map(
        {'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7, 'HS-grad': 8,
         'Prof-school': 9, 'Assoc-acdm': 10, 'Assoc-voc': 11, 'Some-college':12, 'Bachelors': 13, 'Masters': 14,'Doctorate': 15}).astype(int)

    #create quickaccess list with numerical variables labels
    numvars = ['age', 'hours-per-week']

    #  dic categories Index(['A11', 'A12', 'A13', 'A14'], dtype='object')
    dict_categorical = {}
    for c in catvars:
        dict_categorical[c] = list(df[c].astype("category").cat.categories)
        df[c] = df[c].astype("category").cat.codes

    #  all features as float
    df = df.astype("float64")

    df["income"] = df["income"].astype("int32")
    # save codes
    with open('dict_adult.txt', 'w') as f:
        f.write(str(dict_categorical))

    return df[nodes]

### START EXPERIMENT
'''
# Downoload and save the dataset
s = requests.get("https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv").text

df = pd.read_csv(io.StringIO(s), names=["age", "workclass", "fnlwgt", "education", "education-num",
                                         "marital-status", "occupation", "relationship", "ethnicity", "gender",
                                         "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"])

# save original dataset
df.to_csv("data/adult_income_dataset.csv")  # save as csv file
'''

#  generate graph
G = graph_adult_income()

nodes = list(G.nodes)

# info about features
constraints_features = {"immutable": ["ethnicity", "native-country", "gender"], "higher": ["age", "education"]}
categ_features = ["gender", "ethnicity", "occupation", "marital-status", "education", "workclass", "relationship", "native-country"]

# load dataset
df = load_adultdataset(nodes)

immutable_features = ["ethnicity", "native-country", "gender", "age"]


descriptions = {
    "ethnicity": "Refers to an individual's ethnic lineage. Helps in understanding socio-economic patterns and identifying potential discrimination. Value example: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black",

    "income": "The derived feature representing an individual's estimated annual income, categorized as <=50K or >50K.",

    "occupation": "Denotes a person's job role. Key for understanding income variation based on professional domains. Value example: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.",

    "marital-status": "Represents marital standing. Useful for insights on combined incomes, financial commitments, and fiscal stability. Value example: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.",

    "hours-per-week": "The number of work hours weekly. Directly relates to earnings potential and employment type.",

    "education": "Denotes academic level. Highlights the relationship between education, job prospects, and earnings. Value example: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool",

    "age": "Indicates an individual's age. Offers insights into career stage and potential earnings.",

    "workclass": "The employment status of the individual, categorized into Private, Self-emp, Govt, Without-pay, or Never-worked. This feature is valuable for predicting annual income by highlighting income disparities, occupation types, job stability, and interactions with other features like education, aiding in more accurate income predictions.",

    "relationship": "Outlines family dynamics, like 'Wife' or 'Unmarried'. Helps understand household financial responsibilities. Value example: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried",

    "native-country": "Individual's birth country. Offers insights into economic backgrounds and potential income based on origin.",

    "gender": "Specifies as Female or Male. Useful for highlighting potential income disparities and gender-based biases."
}

llm = ChatOpenAI(temperature= 0, model="gpt-4")

result = get_graphs(df, descriptions, immutable_features, "individual's annual income results from various factors", "income", "results", llm)