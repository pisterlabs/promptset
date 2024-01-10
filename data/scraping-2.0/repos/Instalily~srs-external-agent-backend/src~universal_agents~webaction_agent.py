
import os
import dotenv
import json
from openai import AsyncAzureOpenAI
from typing import Optional, List
from pydantic import BaseModel, Field, RootModel
from src.universal_agents.prompts import WEBACTION_PROMPT
from src.shopping_flow.agents.constants import PRIMARY_EXCLUDED_FIELDS
from src.logging.cache import recommended_product_links

dotenv.load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ACTION_AGENT_AZURE_API_KEY = os.environ.get("ACTION_AGENT_AZURE_API_KEY")
class Action(BaseModel): 
    ai_chat_response: str = Field(description="AI Response")
    action: str = Field(description="Action to be taken. Can be 'jumpToURL', 'click', 'forward', or 'back'")
    args: str = Field(description='Args parameter for corresponding action')

class AgentResponse(RootModel): 
    root: List[Action] = Field(description="Sequence of actions to be taken on the webpage in order of their occurrence in the list")

class WebActionAgent:
    def __init__(self, user_id) -> None:
        self.client = AsyncAzureOpenAI(
            api_key=ACTION_AGENT_AZURE_API_KEY,
            api_version="2023-10-01-preview",
            azure_endpoint = "https://action-agents.openai.azure.com/"
            )
        self.user_id = user_id
        
    def truncate_product_details(self, product_details: List[dict]) -> List[dict]:
        """
        Truncates the product details (list of dictionaries, with each dictionary corresponding to the details of a unique product) that are then appended to the action agent's prompt when product context is needed. 
        """
        # Just take the first recommendation
        try:
            product = product_details[0]
            new_product_dic = {k: v for k, v in product.items() if (k not in set(PRIMARY_EXCLUDED_FIELDS) and v is not None)}
            return new_product_dic
        except Exception as e:
            print("ERROR AT ACTION AGENT", e)
            
    async def execute(
        self, 
        chat_history: str, 
        current_url: str, 
        product_context: Optional[List[dict]]= None
    ) -> List[dict]:
        product_links = recommended_product_links[self.user_id]
        action_prompt = WEBACTION_PROMPT
        action_prompt += f"Chat history:\n{chat_history}\nCurrent URL:\n {current_url}\nProduct links: {product_links}"
        
        # If we're given product details we need to provide them in the prompt
        if product_context:
            truncated_product_context = self.truncate_product_details(product_context)
            action_prompt += f"\nProduct Context:\n{truncated_product_context}"
    
        agent_response = await self.client.chat.completions.create(
            model="agent-api",
            stream=False,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": action_prompt}],
        )
        output_message = agent_response.choices[0].message.content

        try:
            json_output = json.loads(output_message)
            agent_output = json_output["output"]
            return agent_output
            # verified_output = await self.verify_output(agent_output)
            # return verified_output
        except Exception as e:
            try:
                parsed_data = AgentResponse.model_validate_json(output_message)
                agent_sequences = parsed_data.root
                sequences_dict = [dict(item) for item in agent_sequences]
                return sequences_dict
            except Exception as exc:
                print('JSON parsing during fn execute_agent file webaction_agent.py failed:', e)
                print('Pydantic parsing during fn execute_agent file webaction_agent.py failed:', exc)
                return None
        
    async def verify_output(self, action_agent_ouput: List[dict]) -> List[dict]:
        product_links = recommended_product_links[self.user_id]
        CHECK_PROMPT = """You're given a sequence of actions as well as a dictionary of products mapped to their URL. Double check that if any of the actions involves navigating to a product's URL that the correct url is provided. 
        Your output should be the exact same as what was provided except for the necessary changes. 
        Output strictly in JSON format in sequential order of actions:
        {"output":[\{{"ai_chat_response":"str", "action":"Click"/"jumpUrl", "args": link or element to click\}}, additional actions as needed]}
        """
        CHECK_PROMPT += f"\nAction sequence: {action_agent_ouput}\nProduct URLs: {product_links}"
        agent_response = await self.client.chat.completions.create(
            model="agent-api",
            stream=False,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": CHECK_PROMPT}],
        )
        output_message = agent_response.choices[0].message.content

        try:
            json_output = json.loads(output_message)
            agent_output = json_output["output"]
            return agent_output
        except Exception as e:
            try:
                parsed_data = AgentResponse.model_validate_json(output_message)
                agent_sequences = parsed_data.root
                sequences_dict = [dict(item) for item in agent_sequences]
                return sequences_dict
            except Exception as exc:
                print('JSON parsing during fn execute_agent file product_agent.py failed:', e)
                print('Pydantic parsing during fn execute_agent file product_agent.py failed:', exc)
                return None
            


   