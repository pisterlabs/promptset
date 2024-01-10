import re
import json
import requests
# Import things that are needed generically
from langchain.tools import BaseTool, Tool
from typing import Optional
from llms.azure_llms import create_llm
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate, LLMChain
from tools.borealis_tools import extract_ips_urls_domains

tool_llm = create_llm(temp=0.4)
tool_llm_temp0 = create_llm(temp=0)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

info_template = """Extract data from json into a readable report, include all ips, domains and numerical information if there is any: {info}"""
info_prompt_template = PromptTemplate(input_variables=["info"], template=info_template)
answer_chain = LLMChain(llm=tool_llm, prompt=info_prompt_template)
        
def queryOpenCTI(searchIP):
    # Define the GraphQL endpoint URL
    url = 'https://opencti.collaboration.cyber.gc.ca/graphql?'
    # Set the headers for the request
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ae69080b-5047-469f-852c-f8c01c794a7b',
    }
    var1 = searchIP
    # Define the GraphQL query with the variable inserted
    query = '''
        query($search: String!) {
        stixCyberObservables(search: $search) {
            edges {
            node {
                entity_type
                id
                observable_value
                created_at
                updated_at
                x_opencti_score
                createdBy {
                id
                name
                }
                stixCoreRelationships {
                edges {
                    node {
                    fromType
                    fromId
                    from {
                        __typename
                        ... domainNameFragment
                        ... ipv4Fragment
                        ... malwareFragment
                        ... textFragment
                    }
                    entity_type
                    relationship_type
                    confidence
                    toId
                    to {
                        __typename
                        ... domainNameFragment
                        ... ipv4Fragment
                        ... malwareFragment
                        ... textFragment
                    }
                    }
                }
                }
            }
            }
        }
        stixCoreRelationship(id: "e62ca35f-dfc5-4b43-a905-8fece8572cd6") {
            createdBy {
            __typename
            id
            entity_type
            name
            }
        }
        }

        fragment domainNameFragment on DomainName {
        value
        }

        fragment ipv4Fragment on IPv4Addr {
        value
        }

        fragment malwareFragment on Malware {
        name
        }

        fragment textFragment on Text {
        value
        }
    '''
    payload = {
        'query': query,
        'variables': {
            'search': var1
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return str(response.json())
    else:
        print('Request failed with status code:', response.status_code)
        
def get_openCTI_response(query_list):
    data = []
    for elem in query_list:
        data.append(answer_chain.run(queryOpenCTI(elem)))
    return "\n".join(data)

def openCTI_search_processing(query):
    ips, urls, domains = extract_ips_urls_domains(query)
    if len(ips) > 0 or len(domains) > 0:
        query_list = ips+domains
        response = get_openCTI_response(query_list)
        return response
    else:
        return None
    
class openCTI_tool(BaseTool):
    name = "Associated IP Lookup"
    description = "use for getting IPs associated with a domain or IP"
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        try:
            return openCTI_search_processing(query)
        except Exception as e:
            return str(e)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    
openCTI_lookup = openCTI_tool()
openCTI_tool = Tool(
    name = "Associated IP Lookup",
    description = "use for getting IPs associated with a domain or IP",
    func= openCTI_lookup.run
)
