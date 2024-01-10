################################################################################
# Importing relevant packages
################################################################################
import os
from typing import Any, Dict, List
import re
import json
from configparser import ConfigParser
import openai
from neo4j import GraphDatabase
from openai import OpenAI

################################################################################
# Execute Graph Operations
################################################################################
def refine_query(client,user_query):
    """Summary: 
        This function is used to refine the user query so that the LLM can
        return a coherant result.
    ---------------------------------------------------------------------------
    Args:
        user_query :- Query given by the user
    ---------------------------------------------------------------------------
    Returns:
        json file with the refined query or error message
    """
    # Set up OpenAI API key
    openai.api_key = os.environ['OPENAI_API_KEY']

    # Get user input
    raw_user_input = user_query

    # Constructing the prompt for the LLM to refine the user input
    prompt_to_refine = (
        "The following is a plain English user input for querying a graph database: '{}'. "
        "Rewrite this input into a clearer query format. "
        "However, do not remove any information related to the database or information related to node type which the user provided. If the input is not clear or relevant, indicate that the user should provide a clearer query. "
        "Do not remove any information related to database and node attributes."
        "For example, if the input is 'I want to know how many nodes are in this database', "
        "a better format would be 'Find the number of nodes in the database'.\n"
        "Refined Input: ".format(raw_user_input)
    )

    max_tokens = 100
    if len(raw_user_input.split()) > 100:
        max_tokens = len(raw_user_input.split())

    try:
        # Requesting OpenAI API to refine the prompt
        response = client.completions.create(
            model="text-davinci-003",  # Use the latest available engine
            prompt=prompt_to_refine,
            max_tokens=max_tokens  # Adjust based on your needs
        )

        refined_prompt = response.choices[0].text.strip()

        # Check if the refined prompt is valid
        if "provide a clearer query" in refined_prompt.lower():
            return {'statusCode': 400, 'body': json.dumps(output["updated_user_prompt"])}
        else:
            output = {"statusCode": 200, "updated_user_prompt": refined_prompt}
            updated_user_query = output["updated_user_prompt"]
            print("Refined Output:\n", updated_user_query)
            return output

    except Exception as err:
        print(f"Error in refining prompt: {err}")
        return {'statusCode': 400,'body': str(err)}

def neo4j_query(query: str, driver , params: dict = {}) -> List[Dict[str, Any]]:
    """
    Summary:
        Function to execute a neo4j query and return the results
    ---------------------------------------------------------------------------
    Args:
        query :- Neo4j cypher query
        driver :- neo4j driver
        params :- extra params for neo4j
    ---------------------------------------------------------------------------
    Returns:
        Neo4j output 
    """
    from neo4j.exceptions import CypherSyntaxError

    with driver.session(database="neo4j") as session:
        try:
            data = session.run(query, params)
            return [r.data() for r in data]
        except CypherSyntaxError as err:
            raise ValueError(f"Generated Cypher Statement is not valid\n{err}")
          
def refresh_schema(driver) -> str:
    """
    Summary:
        Refreshes the Neo4j graph schema information.
    ---------------------------------------------------------------------------
    Args:
        driver :- Neo4j driver
    ---------------------------------------------------------------------------
    Returns:
        A string which describes the schema of the graph
    """
    
    # Query to get properties of nodes
    node_properties_query = """
    CALL apoc.meta.data()
    YIELD label, other, elementType, type, property
    WHERE NOT type = "RELATIONSHIP" AND elementType = "node"
    WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
    RETURN {labels: nodeLabels, properties: properties} AS output

    """

    # Query to get properties of relationships
    rel_properties_query = """
    CALL apoc.meta.data()
    YIELD label, other, elementType, type, property
    WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
    WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
    RETURN {type: nodeLabels, properties: properties} AS output
    """

    # Query to get node and relationship properties
    rel_query = """
    CALL apoc.meta.data()
    YIELD label, other, elementType, type, property
    WHERE type = "RELATIONSHIP" AND elementType = "node"
    UNWIND other AS other_node
    RETURN {start: label, type: property, end: toString(other_node)} AS output
    """
    
    node_properties = [el["output"] for el in neo4j_query(node_properties_query,driver)]
    rel_properties = [el["output"] for el in neo4j_query(rel_properties_query,driver)]
    relationships = [el["output"] for el in neo4j_query(rel_query,driver)]

    schema_str = f"""
    Node properties are the following:
    {node_properties}
    Relationship properties are the following:
    {rel_properties}
    The relationships are the following:
    {[f"(:{el['start']})-[:{el['type']}]->(:{el['end']})" for el in relationships]}
    """
    
    return schema_str

def get_gpt3_response(curr_schema, question, client, model="gpt-3.5-turbo",history=None):
    """
    Send a request to the OpenAI Chat API and get a response from the model to
    obtain the relevant cypher query
    ---------------------------------------------------------------------------
    Args:
        curr_schema : Schema of the graph described as a string
        question : Question entered by the user
        client : OpenAI client
        model : The model version to use, default is "gpt-3.5-turbo".
        history: Historical openai chat responses
    ---------------------------------------------------------------------------
    Returns:
        str: The model's response.
    """
    system_prompt =  """
    
    "Human: Task:Generate Cypher statement to query a graph database.\nInstructions:\nUse only the provided 
    relationship types and properties in the schema.\nDo not use any other relationship types or 
    properties that are not provided.\n
    Consider directionality of the graph.\n
    The cypher output should have some indication either as variable name to indicate the requirement of the 
    question.\n
    Do not include any explanations or apologies in your responses.\n
    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
    \nSchema:
    {}".
    """.format(curr_schema)
    
    # Create the full prompt by combining the system prompt, context, and the user question
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    if history:
        messages.extend(history)
    
    ## Use the OpenAI Python client to send the request
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    # Extract the response text and return
    return response.choices[0].message.content

def get_gpt3_response_2(prompt, question, client,model="gpt-3.5-turbo",history=None):
    """
    Send a request to the OpenAI Chat API and get a response from the model to
    obtain the relevant user output based on neo4j output.
    ---------------------------------------------------------------------------
    Args:
        prompt : Output from neo4j
        question : Question entered by the user
        client : OpenAI client
        model : The model version to use, default is "gpt-3.5-turbo".
        history: Historical openai chat responses
    ---------------------------------------------------------------------------
    Returns:
        str: The model's response.
    """
    system_prompt =  """
    
    "Human: You are an assistant that helps to form nice and human understandable answers
    .\nThe information part contains the provided information that you must use to construct an answer.
    \nThe provided information is authoritative, you must never doubt it or try to use your internal knowledge
    to correct it.\nMake the answer sound as a response to the question. 
    Do not mention that you based the result on the given information.\n
    Output only the node ids to represent the nodes, unless specified
    \nInformation:\n{}"
    """.format(prompt)
    
    # Create the full prompt by combining the system prompt, context, and the user question
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    if history:
        messages.extend(history)
    
    # Use the OpenAI Python client to send the request
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    # Extract the response text and return
    return response.choices[0].message.content


def lang_chain_custom(question,client,driver,model="gpt-4"):
    """
    Summary:
        Obtain an answer to the user question based on the graph
    ---------------------------------------------------------------------------
    Args:
        question : Question entered by the user
        client : OpenAI client
        driver: Neo4j driver
        model : The model version to use, default is "gpt-4".
    ---------------------------------------------------------------------------
    Returns:
        json file with the required output or error message
    """
    try:
        # Obtaining the graph schema
        curr_schema = refresh_schema(driver)  
        
        # Getting Cypher query
        response = get_gpt3_response(curr_schema, question,client, model)
        print('Cypher Query is',response)
    except Exception as err:
        return {'statusCode': 400, 'body': str(err)} 
    
    # Interacting with neo4j
    try:
        cypher_response = neo4j_query(response,driver)
        cypher_response = cypher_response[:10] ## Limiting it to avoid token limits
        print('Neo4j answer is',cypher_response)
    except Exception as err:
        try:
            print('Retrying')
            history = [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": f"""This query returns an error: {str(err)} 
                        Give me a improved query that works without any explanations or apologies""",
                    },
            ]
            response = get_gpt3_response(curr_schema, question,client,model,history)
            print('New Cypher Query is',response)
            cypher_response = neo4j_query(response,driver)
            cypher_response = cypher_response[:10] ## Limiting it to avoid token limits
            print('Neo4j answer is',cypher_response)
        except Exception as err:
            #output = {"status": "failed", "updated_user_prompt": refined_prompt}
            return {'statusCode': 400, 'body': str(err)}
    
    # Obtaining final answer
    try:
        response_2 = get_gpt3_response_2(cypher_response, question,  client, model)
        
        pattern = r'\bsorry\b'
        # re.IGNORECASE flag makes the search case-insensitive
        if(bool(re.search(pattern, response_2, re.IGNORECASE))):
            
            print('Retrying output string')
            
            history = [
            {"role": "assistant", "content": response_2},
            {
                "role": "user",
                "content": f"""This query returns an incomplete answer.
                If the information is empty then return does not exist or unknown.\n  
                If not, assume the output is a simplified answer to 
                the question {question} 
                Give me a improved query that works without any explanations or apologies""",
            },

            ]
        
            response_2 = get_gpt3_response_2(cypher_response, question, client, model,history)
    except Exception as err:
        return {'statusCode': 400, 'body': str(err)}
    
    return {'statusCode': 200, 'body': json.dumps(response_2)}

def execute_graph_operations(config_path: str, user_query: str, network_choice: str) -> Dict:
    """
    Summary: This function connects to the Neo4j DB, creates Cypher
    Queries and executes it on Neo4j DB
    ----------------------------------------------------------------------
    Extra args:
    file_path: Is a Path object which points to the file with data
    ----------------------------------------------------------------------
    """
    # Get configuration
    configur = ConfigParser()
    configur.read(config_path)

    try:
        uri = None
        username = None
        password = None
        if network_choice == "BGP Networking Data":
            # Get neo4j credentials
            uri = configur.get('bgp-graph', 'uri')
            username = configur.get('bgp-graph', 'username')
            password = configur.get('bgp-graph', 'password')
        else:
            # Ask user to input the dataset
            return {'statusCode': 400,
                'body': json.dumps("This dataset is not available. Please use our \"Get Data\" functionality.")}
    except Exception as err:
        print("Error in getting Graph DB credentials")
        print(err)
        return {'statusCode': 400,
                'body': str(err)}

    print(uri)
    print(username)
    print(password)

    # Connecting to Graph DB
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))

    except Exception as err:
        print("Error in connecting to Graph DB")
        print(err)
        return {'statusCode': 400,
                'body': str(err)}

    # Set Up OpenAI API and get response
    try:
        api_key = configur.get('openai-api', 'api-key')
        name = 'OPENAI_API_KEY'
        os.environ[name] = api_key
        client = OpenAI()
        
        ## Updating the user query        
        refined_output = refine_query(client=client,user_query=user_query)
        
        if(refined_output['statusCode']==400):
            return refined_output
        else:
            updated_user_query = refined_output["updated_user_prompt"]
            # Calling custom Langchain 
            response = lang_chain_custom(question=updated_user_query,
                                        client=client,
                                        driver=driver,
                                        model="gpt-4")
            return response
    except Exception as err:
        print(str(err))
        print("Error in executing Graph Operation.")
        return {'statusCode': 400,
                'body': str(err)}
