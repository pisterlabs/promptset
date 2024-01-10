from langchain.tools import BaseTool
from typing import Any
import os
from dotenv import load_dotenv

# Import the OperatorSearchAPI and Query classes
from operatorio import OperatorSearch, Query, EntityType

load_dotenv()

api_key = os.environ.get("OPERATOR_API_KEY")

class OperatorTool(BaseTool):
    name = "Operator Address Search"
    description = '''
    Use this tool when you need to find the right address for a specific project. 
    Possible options for entity_type are nft, identity, wallet, and token.
    Use the full name of the blockchain, i.e. "Ethereum" instead of "ETH" for the blockchain parameter.
    Default to Ethereum if no blockchain is specified. Search for names as is, do not attempt to interpret spelling errors.
    '''

    def _run(self, search: str, blockchain: str = "Ethereum", entity_type: EntityType = EntityType.token):
        
        # Initialize the OperatorSearchAPI with the retrieved API key
        api = OperatorSearch(api_key)

        # Define a query to search for the task
        query = Query(
            query=search,
            blockchain=blockchain,
            entity_type=entity_type,
            query_by=[]
        )

        # Use the OperatorSearchAPI to perform the search
        entities = api.search(query)

        # Return the top result's address
        return entities.matches[0].address if entities.matches else None

    def _arun(self, task: Any):
        raise NotImplementedError("This tool does not support async")
