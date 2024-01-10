!pip install langchain.agents
import os
from itertools import combinations

import numpy as np
from scipy import linalg 
from scipy import stats 

import matplotlib.pyplot as plt

from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

from langchain.chat_models import ChatOpenAI

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.algorithms import PC

from castle.common.priori_knowledge import PrioriKnowledge


COLORS = [
    '#00B0F0',
    '#FF0000',
    '#B0F000'
]



def check_if_dag(A):
    return np.trace(linalg.expm(A * A)) - A.shape[0] == 0




#Put in your OPENAI KEY HERE
os.environ['OPENAI_API_KEY'] = "sk-KEYASGFASGUASOGUHA"

all_vars = {
    'altitude': 0,
    'oxygen_density': 1,
    'temperature': 2,
    'risk_of_death': 3,
    'mehendretex': 4
}


SAMPLE_SIZE = 1000

altitude = stats.halfnorm.rvs(scale=2000, size=SAMPLE_SIZE)
temperature = 25 - altitude / 100 + stats.norm.rvs(
    loc=0,
    scale=2,
    size=SAMPLE_SIZE
)

mehendretex = stats.halfnorm.rvs(size=SAMPLE_SIZE)

oxygen_density = np.clip(
    1 - altitude / 8000 
    - temperature / 50 
    + stats.norm.rvs(size=SAMPLE_SIZE) / 20,
    0, 
    1)

risk_of_death = np.clip(
    altitude / 20000 
    + np.abs(temperature) / 100 
    - oxygen_density / 5 
    - mehendretex / 5
    + stats.norm.rvs(size=SAMPLE_SIZE) / 10,
    0,
    1
)



dataset = np.stack(
    [
        altitude,
        oxygen_density,
        temperature,
        risk_of_death,
        mehendretex
    ]
).T



true_dag = np.array(
    [
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ]
)


# PC discovery without LLM assist
pc = PC(variant='stable')
pc.learn(dataset)

print(f'Is learned matrix a DAG: {check_if_dag(pc.causal_matrix)}')

# Vizualize
GraphDAG(
    est_dag=pc.causal_matrix, 
    true_dag=true_dag)

plt.show()

# Compute metrics
metrics = MetricsDAG(
    B_est=pc.causal_matrix, 
    B_true=true_dag)

print(metrics.metrics)

# Instantiate and encode priori knowledge
priori_knowledge = PrioriKnowledge(n_nodes=len(all_vars))

llm = ChatOpenAI(
    temperature=0, # Temp == 0 => we want clear reasoning
    model='gpt-4')#'gpt-3.5-turbo') 


# Load tools
tools = load_tools(
    [
        "wikipedia"
    ], 
    llm=llm)


# Instantiate the agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False)



def get_llm_info(llm, agent, var_1, var_2):
    
    out = agent(f"Does {var_1} cause {var_2} or the other way around?\
    We assume the following definition of causation:\
    if we change A, B will also change.\
    The relationship does not have to be linear or monotonic.\
    We are interested in all types of causal relationships, including\
    partial and indirect relationships, given that our definition holds.\
    ")
    
    print(out)
    
    pred = llm.predict(f'We assume the following definition of causation:\
    if we change A, B will also change.\
    Based on the following information: {out["output"]},\
    print (0,1) if {var_1} causes {var_2},\
    print (1, 0) if {var_2} causes {var_1}, print (0,0)\
    if there is no causal relationship between {var_1} and {var_2}.\
    Finally, print (-1, -1) if you don\'t know. Importantly, don\'t try to\
    make up an answer if you don\'t know.')
    
    print(pred)
    
    return pred

for var_1, var_2 in combinations(all_vars.keys(), r=2):
    print(var_1, var_2)
    out = get_llm_info(llm, agent, var_1, var_2)
    if out=='(0,1)':
        priori_knowledge.add_required_edges(
            [(all_vars[var_1], all_vars[var_2])]
        )
        
        priori_knowledge.add_forbidden_edges(
            [(all_vars[var_2], all_vars[var_1])]
        )

    elif out=='(1,0)':
        priori_knowledge.add_required_edges(
            [(all_vars[var_2], all_vars[var_1])]
        )
        priori_knowledge.add_forbidden_edges(
            [(all_vars[var_1], all_vars[var_2])]
        )

print('\nLLM knowledge vs true DAG')
priori_dag = np.clip(priori_knowledge.matrix, 0, 1)

print(f'\nChecking if priori graph is a DAG: {check_if_dag(priori_dag)}')

GraphDAG(
    est_dag=priori_dag, 
    true_dag=true_dag)

plt.show()

print('\nRunning PC')

# Instantiate the model with expert knowledge
pc_priori = PC(
    priori_knowledge=priori_knowledge,
    variant='stable'
)

# Learn
pc_priori.learn(dataset)

GraphDAG(
    est_dag=pc_priori.causal_matrix, 
    true_dag=true_dag)

plt.show()

# Compute metrics
metrics = MetricsDAG(
    B_est=pc_priori.causal_matrix, 
    B_true=true_dag)

print(metrics.metrics)

import numpy as np

# Define the order of variables
variables_order = ["accommodates", "price", "bedrooms", "beds", "bathrooms"]

# Initialize the adjacency matrix with zeros
n_vars = len(variables_order)
true_dag = np.zeros((n_vars, n_vars))

# Define the relationships as given in the digraph format
relationships = [
    ("accommodates", "price"),
    ("bedrooms", "accommodates"),
    ("beds", "bedrooms"),
    ("bathrooms", "bedrooms"),
    ("bathrooms", "price")
]

# Fill in the matrix based on the relationships
for source, target in relationships:
    i = variables_order.index(source)
    j = variables_order.index(target)
    true_dag[i][j] = 1

print(true_dag)



import os
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.algorithms import PC

import os
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.algorithms import PC
from castle.common.priori_knowledge import PrioriKnowledge

# Load the dataset
file_path = r"C:\Users\nstep\TSU\SeniorProject\df_selected1.csv"
nashvilleDF = pd.read_csv(file_path)
data_subset = nashvilleDF[['accommodates', 'price','bedrooms','beds','bathrooms']].dropna().values

# Define the order of variables and create the true_dag
variables_order = ["accommodates", "price", "bedrooms", "beds", "bathrooms"]
all_vars = {var: idx for idx, var in enumerate(variables_order)}
n_vars = len(variables_order)
true_dag = np.zeros((n_vars, n_vars))
relationships = [
    ("accommodates", "price"),
    ("bedrooms", "accommodates"),
    ("beds", "bedrooms"),
    ("bathrooms", "bedrooms"),
    ("bathrooms", "price")
]
for source, target in relationships:
    i = variables_order.index(source)
    j = variables_order.index(target)
    true_dag[i][j] = 1

# Run PC discovery without LLM assist
pc = PC(variant='stable')
pc.learn(data_subset)

# Visualize the result
GraphDAG(
    est_dag=pc.causal_matrix, 
    true_dag=true_dag)

plt.show()

# Compute metrics
metrics = MetricsDAG(
    B_est=pc.causal_matrix, 
    B_true=true_dag)
print(metrics.metrics)

# Instantiate and encode priori knowledge
priori_knowledge = PrioriKnowledge(n_nodes=len(all_vars))

# Instantiate the GPT agent (You'll need to provide the appropriate setup and imports for the LLM)
# This step is based on the code you provided
llm = ChatOpenAI(
    temperature=0, # Temp == 0 => we want clear reasoning
    model='gpt-4')#'gpt-3.5-turbo')

tools = load_tools(
    ["wikipedia"], 
    llm=llm)

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False)

def get_llm_info(llm, agent, var_1, var_2):
    
    out = agent(f"Does {var_1} cause {var_2} or the other way around?\
    We assume the following definition of causation:\
    if we change A, B will also change.\
    The relationship does not have to be linear or monotonic.\
    We are interested in all types of causal relationships, including\
    partial and indirect relationships, given that our definition holds.\
    ")
    
    print(out)
    
    pred = llm.predict(f'We assume the following definition of causation:\
    if we change A, B will also change.\
    Based on the following information: {out["output"]},\
    print (0,1) if {var_1} causes {var_2},\
    print (1, 0) if {var_2} causes {var_1}, print (0,0)\
    if there is no causal relationship between {var_1} and {var_2}.\
    Finally, print (-1, -1) if you don\'t know. Importantly, don\'t try to\
    make up an answer if you don\'t know.')
    
    print(pred)
    
    return pred

# Add priori knowledge from the LLM
for var_1, var_2 in combinations(all_vars.keys(), r=2):
    print(var_1, var_2)
    out = get_llm_info(llm, agent, var_1, var_2)
    if out=='(0,1)':
        priori_knowledge.add_required_edges([(all_vars[var_1], all_vars[var_2])])
        priori_knowledge.add_forbidden_edges([(all_vars[var_2], all_vars[var_1])])
    elif out=='(1,0)':
        priori_knowledge.add_required_edges([(all_vars[var_2], all_vars[var_1])])
        priori_knowledge.add_forbidden_edges([(all_vars[var_1], all_vars[var_2])])

# Check the priori knowledge
print('\nLLM knowledge vs true DAG')
priori_dag = np.clip(priori_knowledge.matrix, 0, 1)
print(f'\nChecking if priori graph is a DAG: {check_if_dag(priori_dag)}')

GraphDAG(
    est_dag=priori_dag, 
    true_dag=true_dag)
plt.show()

# Now, run the PC algorithm with priori knowledge
print('\nRunning PC with Priori Knowledge')

# Instantiate the model with expert knowledge
pc_priori = PC(
    priori_knowledge=priori_knowledge,
    variant='stable'
)

# Learn using the dataset
pc_priori.learn(data_subset)

# Visualize
GraphDAG(
    est_dag=pc_priori.causal_matrix, 
    true_dag=true_dag)
plt.show()

# Compute metrics
metrics_priori = MetricsDAG(
    B_est=pc_priori.causal_matrix, 
    B_true=true_dag)
print(metrics_priori.metrics)
