import os
import json
from openai import AzureOpenAI, AsyncAzureOpenAI
from src.shopping_flow.agents.constants import GATE_RESPONSE_MAPPING
from src.shopping_flow.agents.prompts import GATING_PROMPT
from timeit import default_timer as timer

import dotenv
dotenv.load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

ACTION_AGENT_AZURE_API_KEY = os.environ.get("ACTION_AGENT_AZURE_API_KEY")
class GateAgent():
    """
    Custom agent that uses the guidance function to determine if a message is a SQL NLQ
    """
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=ACTION_AGENT_AZURE_API_KEY,
            api_version="2023-10-01-preview",
            azure_endpoint = "https://action-agents.openai.azure.com/"
            )
       
    async def execute(self, chat_history: str, user_query: str) -> str:
        """
        Entry point for gate agent. Determines whether or not the current chat history is relevant to shopping flow and returns the appropriate response. An empty string response means the shopping flow should be initiated.
        """
        return await self.determine_query_category(chat_history, user_query)
    
    async def determine_query_category(
        self,
        chat_history: str,
        user_query: str
    ) -> str:
        """
        Calls the gate agent and outputs a query category from 1 - 4. See prompt for more details.
        """
        start = timer()
        gate_agent_prompt = GATING_PROMPT
        gate_agent_prompt += f"\nChat history: {chat_history}\nUser Query: {user_query}"

        gate_agent_output =  await self.client.chat.completions.create(
            model="agent-api",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": gate_agent_prompt}],
        )
        output_message = gate_agent_output.choices[0].message.content
        json_output = json.loads(output_message)
        query_category = json_output.get("rank", "5")
        
        end = timer()
        print("GATING TIME", end - start)
        print("GATING CATEGORY", query_category)
        return query_category