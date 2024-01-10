# Original ChatGPT ideation tab - needs updating to work

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import streamlit.components.v1 as components
import logging
import openai
import json
import random
import base64
import ast

# Yes yes - move all the non-ui stuff into a controller or something already

CHATGPT_KEY = "chatgpt"

def arrows_uri(input: str | dict) -> str:
    """
    Generates a URI for an arrows app visualization from a json object. WARNING! May overwrite existing arrows drawing.

    Args:
        input: A dictionary or string representing an arrows compatible .json configuration

    Returns:
        A string URI for an arrows app visualization
    """

    # Convert dict to string if needed
    if isinstance(input, dict):
        input = json.dumps(input)

    # Convert the diction object into a base 64 json string
    b = input.encode('utf-8')
    base64_str = base64.b64encode(b).decode('utf-8')

    result = f"https://arrows.app/#/import/json={base64_str}"

    # logging.debug(f'\n\nOutput arrows uri from {input} with base64 JSON: \n{result}')

    return result


def arrows_dictionary(nodes: list[Node], edges: list[Edge], name: str = "GraphGPT Generated Model") -> dict:
    """
    Generates an arrows.app compatible .json file from agraph nodes and edges

    Args:
        nodes: List of agraph Nodes
        edges: List of agraph Edges

    Returns:
        A dictionary matching arrows .json schema
    """
    result_nodes = []
    result_relationships = []
    for n in nodes:
        random_x = round(random.uniform(-600, 600), 14)
        random_y = round(random.uniform(-600, 600), 14)
        result_nodes.append({
            "id": n.id,
            "position":{
                "x" : random_x,
                "y": random_y
            },
            "caption": n.label,
            "style":{},
            "labels":[],
            "properties":{}
        })
    for idx, e in enumerate(edges):
        logging.debug(f'Processing edge to relationships: {e.__dict__}')
        ns = e.source
        nt = e.to
        type = e.label.replace(" ", "_")
        result_relationships.append(
            {
                "id":f"n{idx}",
                "type":type,
                "fromId":ns,
                "toId":nt,
                "style":{},
                "properties":{}
            }
        )
    result = {
        "graph": {
            "style": {
            "font-family": "sans-serif",
            "background-color": "#ffffff",
            "background-image": "",
            "background-size": "100%",
            "node-color": "#ffffff",
            "border-width": 4,
            "border-color": "#000000",
            "radius": 50,
            "node-padding": 5,
            "node-margin": 2,
            "outside-position": "auto",
            "node-icon-image": "",
            "node-background-image": "",
            "icon-position": "inside",
            "icon-size": 64,
            "caption-position": "inside",
            "caption-max-width": 200,
            "caption-color": "#000000",
            "caption-font-size": 50,
            "caption-font-weight": "normal",
            "label-position": "inside",
            "label-display": "pill",
            "label-color": "#000000",
            "label-background-color": "#ffffff",
            "label-border-color": "#000000",
            "label-border-width": 4,
            "label-font-size": 40,
            "label-padding": 5,
            "label-margin": 4,
            "directionality": "directed",
            "detail-position": "inline",
            "detail-orientation": "parallel",
            "arrow-width": 5,
            "arrow-color": "#000000",
            "margin-start": 5,
            "margin-end": 5,
            "margin-peer": 20,
            "attachment-start": "normal",
            "attachment-end": "normal",
            "relationship-icon-image": "",
            "type-color": "#000000",
            "type-background-color": "#ffffff",
            "type-border-color": "#000000",
            "type-border-width": 0,
            "type-font-size": 16,
            "type-padding": 5,
            "property-position": "outside",
            "property-alignment": "colon",
            "property-color": "#000000",
            "property-font-size": 16,
            "property-font-weight": "normal"
            },
            "nodes":result_nodes,
            "relationships":result_relationships,
            "diagramName": name
        }
    }

    logging.debug(f'\n\nProcessed incoming nodes: {nodes}, edges: {edges} to:\n {result}')
    
    return result


def triples_prompt(prompt: str)-> str:
    # Full prompt string to query openai with and finasse expected response
    full_prompt = f"""
    Given a prompt, extrapolate the most important Relationships. 

    Each Relationship must connect 2 Entities represented as an item list like ["ENTITY 1", "RELATIONSHIP", "ENTITY 2"]. The Relationship is directed, so the order matters.

    Use singular nouns for Entities.

    For example; the prompt: `All birds like to eat seeds` should return: ["Bird", "EATS", "Seed"]

    Limit the list to a maximum of 12 relationships. Prioritize item lists with Entities in multiple item lists. Remove duplicates.

    prompt: {prompt}
    """
    return full_prompt

# def agraph_nodes_edges(response: dict) -> tuple[list[Node], list[Edge]]:
#     """
#     Converts an openai response into agraph nodes and relationships

#     Args:
#         response: Dict result from an open completion call that used the json output option
    
#     Returns:
#         A tuple of agraph nodes in a list and agraph edges in a list

#     Raises:
#         ...
#     """
#     logging.debug(f'agraph_nodes_edges() response recieved: {response}')
#     # Response will be a list of 3 item tuples

#     # Answer should be embedded in the 'data' key.
#     answers = response.get('data', response)

#     if isinstance(answers, list) is False:
#         raise Exception('Answers is not a list')

#     logging.debug(f'Object parsed: {answers}')

#     nodes = set()
#     result_edges = []
#     for item in answers:
#         # Each should be a tuple of 3 items, node-edge-node
#         n1 = item[0]
#         r = item[1]
#         n2 = item[2]

#         # Standardize casing
#         r = r.upper()
#         n1 = n1.title()
#         n2 = n2.title()

#         nodes.add(n1)
#         nodes.add(n2) 

#         edge = Edge(source=n1, target=n2, label=r)
#         result_edges.append(edge)

#     result_nodes = []
#     for node_label in list(nodes):
#         node = Node(id=node_label, label=node_label)
#         result_nodes.append(node)

#     logging.debug(f'Nodes returning: {result_nodes}')
#     logging.debug(f'Edges returning: {result_edges}')

#     return result_nodes, result_edges
# Original version using openai model that DOES NOT support json output
def agraph_nodes_edges(response: str | list) -> tuple[list[Node], list[Edge]]:
    """
    Converts an openai response into agraph nodes and relationships

    Args:
        response: String or list in the format of [["node_id_string", "edge_id_string", "another_node_id_string"],...]
    
    Returns:
        A tuple of agraph nodes in a list and agraph edges in a list

    Raises:
        ...
    """
    logging.debug(f'agraph_nodes_edges() response recieved: {response}')
    # Response will be a list of 3 item tuples

    answers = None

    # Convert to list of lists - if needed
    if isinstance(response, str):
        try:
            answers = json.loads(response)
            logging.info(f'JSON parsed response: {answers}')
        except:
            logging.debug(f'Unable to parse string to list with json.loads. Attempting ast...')
            try: 
                answers = ast.literal_eval(response)
            except:
                logging.debug(f'Unable to parse string to list with ast.literal_eval. String format was unexpected.')
    else:
        answers = response
    
    if isinstance(answers, dict):
        # Sometimes openai will return answers with single key dict (that key could be relationships, data, etc)
        key = list(answers.keys())[0]
        answers = answers.get(key, None)

    logging.debug(f'Processing answers: {answers}')

    if isinstance(answers, list) is False:
        raise Exception(f'Answers is not a list: {answers}')

    nodes = set()
    result_edges = []

    answers_dedupped = [t for t in {tuple(r) for r in answers}]

    logging.debug(f'Answers dedupped: {answers_dedupped}')

    for item in answers_dedupped:
        # Each should be a tuple of 3 items, node-edge-node
        n1 = item[0]
        r = item[1]
        n2 = item[2]

        # Standardize casing
        r = r.upper()
        n1 = n1.title()
        n2 = n2.title()

        nodes.add(n1)
        nodes.add(n2) 

        edge = Edge(source=n1, target=n2, label=r)
        result_edges.append(edge)

    result_nodes = []
    for node_label in list(nodes):
        node = Node(id=node_label, label=node_label)
        result_nodes.append(node)

    logging.debug(f'Nodes returning: {result_nodes}')
    logging.debug(f'Edges returning: {result_edges}')

    return result_nodes, result_edges

@st.cache_data
def generate_openai_response(prompt)-> str:
    # TODO: Make this configurable
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        # model="gpt-4",
        response_format={"type":"json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON"},
            {"role": "user", "content": prompt}
            ]
    )
    # TODO: Validate reponse
    content = response.choices[0].message.content
    logging.debug(f'OpenAI Response: {response}, type: {type(content)}')
    return content
    
def agraph_from_sample(prompt: str):
    # TODO: Pull from a file of samples
    openai_response = '[["Sharks", "eat", "big fish"], ["Big fish", "eat", "small fish"], ["Small fish", "eat", "bugs"]]'
    nodes, edges = agraph_nodes_edges(openai_response)
    config = Config(height=400, width=1000, directed=True)

    if nodes is not None:
        agraph(nodes=nodes, 
            edges=edges, 
            config=config) 

def ideate_ui():

    st.markdown("Use a variation of Varun Shenoy's original [GraphGPT](https://graphgpt.vercel.app) to convert a natural language description into a graph data model")

    # OPENAI TEXTFIELD
    new_open_ai_key = st.text_input(f'OpenAI KEY', type="password", value=st.session_state["OPENAI_API_KEY"])

    # Set openAI key
    openai.api_key = new_open_ai_key

    # Display prompt for user input
    sample_prompt = "Sharks eat big fish. Big fish eat small fish. Small fish eat bugs."
    run_openai = True

    b1, b2 = st.columns(2)
    with b1:
        if st.button('Load Sample', key="graphgpt_sample"):
            st.session_state["SAMPLE_PROMPT"] = sample_prompt
    
    with b2:
        if st.button('Load Sample without OpenAI', key="graphgpt_sample_no_key"):
            st.session_state["SAMPLE_PROMPT"] = sample_prompt
            run_openai = False

    prompt = st.text_area("Prompt", value=st.session_state["SAMPLE_PROMPT"])
    if prompt is None or prompt == "":
        return

    nodes = None
    edges = None

    if run_openai == False:
        # Load vetted response to save on hitting openai for the same thing
        response = [["Sharks", "eat", "big fish"], ["Big fish", "eat", "small fish"], ["Small fish", "eat", "bugs"]]
    else: 
        # Send completion request to openAI
        full_prompt = triples_prompt(prompt)
        response = generate_openai_response(full_prompt) 

    # Convert response to agraph nodes and edges
    try:
        nodes, edges = agraph_nodes_edges(response)
    except Exception as e:
        logging.error(f'Problem converting response to agraph nodes and edges. Error: {e}')
        st.error(f'Problem converting prompt to graph. Please try again or rephrase the prompt')

    # Configure and display agraph
    config = Config(width=1000, height=400, directed=True)
    if nodes is None:
        return
    
    # Display data
    st.write('Graph Viewer')
    agraph(nodes=nodes,  
        edges=edges, 
        config=config)
    
    # For displaying JSON schema. This can be quite long though
    # st.write('JSON Representation')
    # arrows_str = json.dumps(arrows_dict, indent=4)
    # st.code(arrows_str)

    # Prep arrows compatible dictioary for button options
    arrows_dict = arrows_dictionary(nodes, edges)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Edit in Arrows"):
            # Prep arrows compatible json
            uri = arrows_uri(arrows_dict)

            logging.info(f'Arrows URI generated: {uri}')

            st.session_state["ARROWS_URI"] = uri
    with b2:
        if st.button("Push to Generator"):
            st.session_state["ARROWS_DICT"] = arrows_dict