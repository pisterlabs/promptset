import streamlit as st
import streamlit_bd_cytoscapejs
import pandas as pd
import streamlit as st
import datetime
import pandas as pd
import streamlit_bd_cytoscapejs
import openai
import re

import streamlit as st
import pyrebase
import datetime
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(layout="wide")

#examples:
# DataFrame for nodes
# nodes_df = pd.DataFrame({
#     'id': ['quantum_networks', 'node_entanglement', 'no_node_entanglement', 'quantum_memory', 'erbium', 'silicon', 'photonic_chip', 'su8', 'fiber_chip'],
#     'label': ['Quantum Networks', 'Requires Node Entanglement', 'Doesn’t Require Node Entanglement', 'Quantum Memory Platform', 'Erbium Atoms', 'Silicon', 'Photonic Chip', 'SU8', 'Fiber-to-Chip Coupling']
# })

# DataFrame for edges
# edges_df = pd.DataFrame({
#     'id': ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'],
#     'source': ['quantum_networks', 'quantum_networks', 'node_entanglement', 'quantum_memory', 'quantum_memory', 'silicon', 'photonic_chip', 'su8'],
#     'target': ['node_entanglement', 'no_node_entanglement', 'quantum_memory', 'erbium', 'silicon', 'photonic_chip', 'su8', 'fiber_chip']
# })

if "auth_user" not in st.session_state or not st.session_state["auth_user"]:
      switch_page("login")

config = st.secrets
openai.api_key = config.open_api_key

config = st.secrets
firebase = pyrebase.initialize_app(config)
db = firebase.database()



message = """"
You are supposed to help
             with the knowledge management system
                        of a research group. You are given the journal entries of a user. You are supposed to create a network graph from this.

For that I want the nodes in the following format:

# DataFrame for nodes
nodes_df = pd.DataFrame({
    'id': ['quantum_networks', 'node_entanglement', 'no_node_entanglement', 'quantum_memory', 'erbium', 'silicon', 'photonic_chip', 'su8', 'fiber_chip'],
    'label': ['Quantum Networks', 'Requires Node Entanglement', 'Doesn’t Require Node Entanglement', 'Quantum Memory Platform', 'Erbium Atoms', 'Silicon', 'Photonic Chip', 'SU8', 'Fiber-to-Chip Coupling']
})

# DataFrame for edges
edges_df = pd.DataFrame({
    'id': ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'],
    'source': ['quantum_networks', 'quantum_networks', 'node_entanglement', 'quantum_memory', 'quantum_memory', 'silicon', 'photonic_chip', 'su8'],
    'target': ['node_entanglement', 'no_node_entanglement', 'quantum_memory', 'erbium', 'silicon', 'photonic_chip', 'su8', 'fiber_chip']
})

So I want to pandas dataframes. Please give me exactly that format and no other format and no other text, that would be super annoying.

Create a knowledge graph from the following journal entries: """

formatted_messages = []
formatted_messages.append({"role": "system", "content": message})

uid = st.session_state["auth_user"]["localId"]

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

def df_to_elements(nodes_df, edges_df):
    nodes_list = nodes_df.to_dict(orient='records')
    nodes_elements = [{'data': node} for node in nodes_list]
    
    edges_list = edges_df.to_dict(orient='records')
    edges_elements = [{'data': edge} for edge in edges_list]

    return nodes_elements + edges_elements
layout = {
    'name': 'cose',
    'animate': False,
    'refresh': 1,
    'componentSpacing': 100,
    'nodeOverlap': 50,
    'nodeRepulsion': 5000,
    'edgeElasticity': 100,
    'nestingFactor': 5,
    'gravity': 80,
    'numIter': 1000,
    'initialTemp': 200,
    'coolingFactor': 0.95,
    'minTemp': 1.0
}




stylesheet = [
    {
        'selector': 'node',
        'style': {
            'background-color': '#11479e',
            'label': 'data(label)',
            'text-valign': 'center',
            'color': 'white',
            'text-outline-width': 2,
            'text-outline-color': '#11479e'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'width': 3,
            'line-color': '#9dbaea',
            'target-arrow-color': '#9dbaea',
            'target-arrow-shape': 'triangle',
            'line-fill': 'linear-gradient'
        }
    }
]


if st.button("generate graph"):
    own_entries = db.child("journals").child(uid).get().each()
    for e in own_entries:
        formatted_messages.append({"role": "user", "content": e.val()["entry"]})
    formatted_messages.append({"role": "system", "content": """Now please Create me a knowledge graph from the journal entries in the pandas dataframe
                               format described above!"""})
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=formatted_messages)["choices"][0]["message"]["content"]
    # Regex for extracting the edges dataframe
    # Regex for extracting the nodes dataframe
    nodes_pattern = r"nodes_df = pd\.DataFrame\(\{\s+'id': \[([^\]]+)\],\s+'label': \[([^\]]+)\]\s+\}\)"
    edges_pattern = r"edges_df = pd\.DataFrame\(\{\s+'id': \[([^\]]+)\],\s+'source': \[([^\]]+)\],\s+'target': \[([^\]]+)\]\s+\}\)"

    nodes_match = re.search(nodes_pattern, completion)

    went_wrong = False
    if nodes_match:
        # Extracting the ids and labels from the matched groups
        # Using `eval()` to safely convert the string representation of a list into an actual list
        nodes_id = eval(nodes_match.group(1))
        nodes_label = eval(nodes_match.group(2))
        
        # Creating the dataframe
        nodes_df = pd.DataFrame({
            'id': nodes_id,
            'label': nodes_label
        })
    else:
        went_wrong = True
        st.write("something went wrong, try again")

    edges_match = re.search(edges_pattern, completion)
    if edges_match:
        # Extracting the ids and labels from the matched groups
        # Using `eval()` to safely convert the string representation of a list into an actual list
        edges_id = eval(edges_match.group(1))
        edges_source = eval(edges_match.group(2))
        edges_target = eval(edges_match.group(3))
        
        # Creating the dataframe
        edges_df = pd.DataFrame({
            'id': edges_id,
            'source': edges_source,
            'target': edges_target
        })
    else:
        went_wrong = True
        st.write("something went wrong, try again")

    if not went_wrong:
        nodes_df = st.data_editor(nodes_df, num_rows="dynamic")
        edges_df = st.data_editor(edges_df, num_rows="dynamic")

        elements = df_to_elements(nodes_df, edges_df)
        node_id = streamlit_bd_cytoscapejs.st_bd_cytoscape(
            elements,
            layout=layout,
            stylesheet=stylesheet,
            key='quantum_networks_graph'
        )
    