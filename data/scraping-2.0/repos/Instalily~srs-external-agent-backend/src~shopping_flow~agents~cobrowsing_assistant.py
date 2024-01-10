import os
import json
import dotenv
from typing import List, Optional
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from src.shopping_flow.agents.constants import BOOL_MAPPING
from src.shopping_flow.agents.prompts import COBROWSING_PROMPT
from pydantic import BaseModel, Field, RootModel

dotenv.load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ACTION_AGENT_AZURE_API_KEY = os.environ.get("ACTION_AGENT_AZURE_API_KEY")
class AgentSequence(BaseModel):
    agent_type: str = Field(description="Name of Agent - 'product_agent' or 'action_agent'")
    is_wine: Optional[bool] = Field(description="is the product being discussed a wine if product_agent")
    needs_product_context: Optional[bool] = Field(description="indicates if we need product info to take actions on this webpage, needed only for action_agent")

class AgentResponse(RootModel):
    root: List[AgentSequence] = Field(description="Sequence of agents to be invoked in order")

class CoBrowsingAgent:
    def __init__(self) -> None:            
        self.client = AsyncAzureOpenAI(
            api_key=ACTION_AGENT_AZURE_API_KEY,
            api_version="2023-10-01-preview",
            azure_endpoint = "https://action-agents.openai.azure.com/"
        )
        
    async def execute(self, chat_history:str, current_url: str, user_query: str)-> List[dict]:
        """
        Entry point for cobrowsing agent. Given the chat history and current url, the agent returns a list of dictionaries. Each dictionary corresponds to an agent to call and any other appropriate arguments. 
        """
        cobrowsing_assistant_prompt = COBROWSING_PROMPT
        cobrowsing_assistant_prompt += f"\nChat history:{chat_history}\nCurrent URL:{current_url}\nUser Query: {user_query}"
       
        stream = await self.client.chat.completions.create(
            stream=False,
            model="agent-api",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": cobrowsing_assistant_prompt}],
        )
        output_message = stream.choices[0].message.content
        try:
            json_output = json.loads(output_message)
            json_output = json_output["output"]
            json_output = self.ensure_bool(json_output)
            return json_output
        except Exception as e:
            try:
                #TODO: parse BOOLEANS PLEASEEE
                parsed_data = AgentResponse.model_validate_json(output_message)
                agent_sequences = parsed_data.root
                sequences_dict = [dict(item) for item in agent_sequences]
                output_sequence = self.ensure_bool(sequences_dict)

                return output_sequence
            except Exception as exc:
                print('JSON parsing during fn execute_agent file cobrowsing_agent.py failed:', e)
                print('Pydantic parsing during fn execute_agent file cobrowsing_agent.py failed:', exc)
                return None
            
    def ensure_bool(self, agent_sequences: List[dict]) -> List[dict]:
        try:
            bool_mapping = BOOL_MAPPING  # Local variable for faster access
            sequences = []

            for sequence in agent_sequences:
                # Check once if the sequence needs processing
                needs_processing = (
                    "is_wine" in sequence and not isinstance(sequence["is_wine"], bool) or
                    "needs_product_context" in sequence and not isinstance(sequence["needs_product_context"], bool)
                )

                if needs_processing:
                    for key in ["is_wine", "needs_product_context"]:
                        if key in sequence and not isinstance(sequence[key], bool):
                            old_value = sequence[key]
                            sequence[key] = bool_mapping.get(old_value.lower(), old_value)

                sequences.append(sequence)

            return sequences
        except Exception as e:
            print("e")
