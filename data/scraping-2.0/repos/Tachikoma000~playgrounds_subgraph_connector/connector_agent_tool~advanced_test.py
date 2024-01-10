import openai
from llama_index.agent import OpenAIAgent
from base import PlaygroundsSubgraphConnectorToolSpec

"""
ADVANCED TEST 
"""
def advanced_query():
    """
    Run a simple test querying the financialsDailySnapshots from Uniswap V3 subgraph using OpenAIAgent and Playgrounds API.
    """
    # Set the OpenAI API key
    openai.api_key = 'YOUR_OPENAI_API_KEY'
    
    # Initialize the tool specification
    connector_spec = PlaygroundsSubgraphConnectorToolSpec(
        identifier="YOUR_SUBGRAPH_OR_DEPLOYMENT_IDENTIFIER", 
        api_key="YOUR_PLAYGROUNDS_API_KEY", 
        use_deployment_id=False
    )
    
    # Setup agent with the tool
    agent = OpenAIAgent.from_tools(connector_spec.to_tool_list())
    query = """
    {
      financialsDailySnapshots(first: 5, where: {dailyVolumeUSD_gt: 100000}, orderBy: timestamp, orderDirection: desc) {
        id
        timestamp
        totalValueLockedUSD
        dailyVolumeUSD
      }
    }
    """
    response = agent.chat(query)
    print(response)

if __name__ == "__main__":
    advanced_query()

