import requests
import os
import sys
import json

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI,  LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

current_directory = os.path.dirname(os.path.realpath(__file__))
backend_directory = os.path.abspath(os.path.join(current_directory,"..",".."))
sys.path.insert(0, backend_directory)

API_KEY = '1070b58dcadf4b6eabd668dab22cfdca'
URL = "https://api.airstack.xyz/gql"

from core.config import Config
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
MODEL_NAME = Config.MODEL_NAME
LLM = OpenAI(model_name=MODEL_NAME, temperature=0)  

import logging
# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# extract params from question

JUDGEMENT_PROMPT = '''
Assuming you are an expert in keyword extraction, 
your task is to extract key terms from user inquiries that will be used to retrieve blockchain-related information.
These keywords are often wallet addresses, ENS domain names, and the like. 
For example, in the question 'can you check the balance of this address 0x60e4d786628fea6478f785a6d7e704777c86a7c6?', 
the extracted parameter would be '0x60e4d786628fea6478f785a6d7e704777c86a7c6'.
Another question:'Check the recent transfer records of this account 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B',
the extracted parameter would be '0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B'
'''

def get_query_params(x):
    response_schemas = [
      ResponseSchema(name="question", description="question is the problem itself.for example,'Check the recent transfer records of this account 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B.',would be 'Check the recent transfer records of this account 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B.'"),
      ResponseSchema(name="params", description="The parameter extracted from the question, for instance 'can you check the balance of this address 0x60e4d786628fea6478f785a6d7e704777c86a7c6?', would be '0x60e4d786628fea6478f785a6d7e704777c86a7c6'.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=JUDGEMENT_PROMPT+"\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    model = LLM

    _input = prompt.format_prompt(question=x)
    output = model(_input.to_string())
    result = output_parser.parse(output)

    print("INFO:     Judge Result:", result)

    return result


# Headers
HEADERS = {
     "Content-Type": "application/json",
     "authorization": API_KEY
}


# query token balance

def fetch_token_balance(question):

    data = get_query_params(question)
    owner_address = data['params']

    TOKEN_BALANCE = f''' 
    query QB5 {{
      TokenBalances(input: {{filter: {{ owner: {{_eq: "{owner_address}"}}}}, limit: 3, blockchain: ethereum}}) {{
        TokenBalance {{
          amount
          chainId
          id
          lastUpdatedBlock
          lastUpdatedTimestamp
          owner {{
            addresses
          }}
          tokenAddress
          tokenId
          tokenType
          token {{
            name
            symbol
          }}
        }}
      }}
    }}
    '''

    # Make the request
    response = requests.post(URL, headers=HEADERS, json={'query': TOKEN_BALANCE})

    # Parse the response
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=4))  # print the data
        return data
    else:
        return {"error": "Query failed", "status_code": response.status_code}



# fetch_token_balance(question)

# query token transfer

def fetch_transfers_history(question):

    data = get_query_params(question)
    token_address = data['params']

    TOKEN_TRANSFERS = f''' 
    query MyQuery {{
      TokenTransfers(
        input: {{filter: {{tokenAddress: {{_eq: "{token_address}"}}}}, blockchain: ethereum}}
      ) {{
        TokenTransfer {{
          amount
          amounts
          blockNumber
          blockTimestamp
          blockchain
          chainId
          from {{
            addresses
          }}
          to {{
            addresses
          }}
          tokenType
        }}
      }}
    }}
    '''

    # Make the request
    response = requests.post(URL, headers=HEADERS, json={'query': TOKEN_TRANSFERS})

    # Parse the response
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=4))  # print the data
        return data
    else:
        return {"error": "Query failed", "status_code": response.status_code}



# fetch_transfers_history(question)



# query identity

def fetch_identity(identity: str):

    data = get_query_params(identity)
    domain_name = data['params']


    IDENTITY_QUERY = f''' 
    query identity {{
      Wallet(input: {{identity: "{domain_name}", blockchain: ethereum}}) {{
        socials {{
          dappName
          profileName
          profileCreatedAtBlockTimestamp
          userAssociatedAddresses
        }}
        tokenBalances {{
          tokenAddress
          amount
          tokenId
          tokenType
          tokenNfts {{
            contentValue {{
              image {{
                original
              }}
            }}
            token {{
              name
            }}
          }}
        }}
      }}
    }}
    '''

    # Make the request
    response = requests.post(URL, headers=HEADERS, json={'query': IDENTITY_QUERY})

    # Parse the response
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=4))  # print the data
        return data
    else:
        return {"error": "Query failed", "status_code": response.status_code}



# fetch_identity(question)


import json
import requests
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

# Define the GraphQL endpoint and API key
quicknode_endpoint = "https://api.quicknode.com/graphql"
api_key = "QN_0ebeb3e022f541b086412061a6640159"

# Set up the GraphQL client
transport = RequestsHTTPTransport(url=quicknode_endpoint, headers={'x-api-key': api_key}, use_json=True)
client = Client(transport=transport, fetch_schema_from_transport=True)


# query ensname via wallet address
def query_ensname(question):
   
    data = get_query_params(question)
    wallet_address = data['params']

    # Define the GraphQL query to get wallet address
    query = gql("""
    query Query($address: String!) {
        ethereum {
            walletByAddress(address: $address) {
                ensName
            }
        }
    }
    """)

    # Set the address
    variables = {
        "address": wallet_address,
    }

    # Execute the query and print the result
    result = client.execute(query, variable_values=variables)

    print("ensname:", result)

    return result



# query_ensname(question)

# query contract detials via contract address

def query_contract(question):

    data = get_query_params(question)
    contract_address = data['params']

    # Define the GraphQL query to get wallet address
    query = gql("""
    query Query($contractAddress: String!) {
        ethereum {
            contract(contractAddress: $contractAddress) {
                address
                isVerified
                name
                symbol
                supportedErcInterfaces
            }
        }
    }
    """)

    # Set the address
    variables = {
        "contractAddress": contract_address,
    }

    # Execute the query and print the result
    result = client.execute(query, variable_values=variables)

    print("contract details:", result)

    return result


# query_contract(question)

def onchain_info_old_agent(question):

    llm = LLM

    tools = [
        Tool(
            name="token balance info",
            func=fetch_token_balance,
            description="Useful for when you need to check the token balance of a specific address."
        ),
        Tool(
            name="Fetch transfers history",
            func=fetch_transfers_history,
            description="Useful for when you need to examine the transfer records of a specific address.for example:'Check the recent transfer records of this account 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B'"
        ),
        Tool(
            name="ens' address info",
            func=fetch_identity,
            description="Useful for when you need to obtain the identity information associated with a specific ENS name."
        ),
        Tool(
            name="contract info",
            func=query_contract, 
            description="Useful for when you need to examine the details of a contract using its address."
        ),
        Tool(
            name="address' ensname info ",
            func=query_ensname, 
            description="Useful when you want to query the ENS domain name via a specific wallet address.for example:'Please check what is the ENS associated with the account 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B'"
        ),
    ]

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description",verbose=True)
    result = agent.run(question)

    print("INFO:     QUERY ONCHAIN_INFO RESULT:", result)

    return result


# question = "Could you please assist me in looking up the account with the domain name 'vitalik.eth'?"
# question = "Can you help me examine the details of this contract: 0xf2A22B900dde3ba18Ec2AeF67D4c8C1a0DAB6aAC?"
# question = "Please check what is the ENS associated with the account 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B"
# question = "Check the recent transfer records of this account 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B."
# question = "Can you check how many kinds of tokens the address 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B has?"

# onchain_info_agent(question)


# Set up the base template
SUB_AGENT_PROMPT_V1 = """Assume you're a master of on-chain information indexing. 
Your task now is to interpret user queries to understand what information they are seeking. 
Once you've clarified their requirements, you must choose the appropriate tools to retrieve the data. 
If there are no corresponding tools for the type of data the user wishes to query, you should respond, 
'I'm sorry, I can't answer this question at the moment,' and suggest where they can find the required information.s

You can use the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: you should input {input} as param of tool
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""


# Set up the base template
SUB_AGENT_PROMPT_V2 = """Assume you're an investment research analyst with in-depth expertise in the areas of blockchain, web3, and artificial intelligence.
Now, your task is to use this knowledge to answer the upcoming questions in the most professional way. Before responding, carefully consider the purpose of each tool and whether it matches the question you need to answer.
You can use the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Action: the action to take, should be one of [{tool_names}]
Action Input: you should input {input} as the parameter of the selected tool

Begin! Choose the appropriate tool to obtain the answer and provide the observed result as your final response.

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

def onchain_info_agent(question: str):

    llm = LLM

    # Define which tools the agent can use to answer user queries
    tools = [
        Tool(
            name="token balance info",
            func=fetch_token_balance,
            description="Useful for when you need to check the token balance of a specific address."
        ),
        Tool(
            name="transfers history info",
            func=fetch_transfers_history,
            description="Useful for when you need to examine the transfer records of a specific address.for example:'Check the recent transfer records of this account 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B'"
        ),
        Tool(
            name="ens' address info",
            func=fetch_identity,
            description="Useful for when you need to obtain the identity information associated with a specific ENS name."
        ),
        Tool(
            name="contract info",
            func=query_contract, 
            description="Useful for when you need to examine the details of a contract using its address."
        ),
        Tool(
            name="address' ensname info ",
            func=query_ensname, 
            description="Useful when you want to query the ENS domain name via a specific wallet address.for example:'Please check what is the ENS associated with the account 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B'"
        ),
    ]

    prompt = CustomPromptTemplate(
        template=SUB_AGENT_PROMPT_V2,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    ) 

    output_parser = CustomOutputParser()

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    result = agent_executor.run(question)

    print("INFO:     Sub Agent Result:", result)

    return result