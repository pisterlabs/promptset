from src.pinecone.pinecone_uploader import DataProcessing
from typing import List
from openai import AsyncAzureOpenAI
from src.shopping_flow.agents.prompts import CONTEXT_PROMPT
import os
import dotenv

dotenv.load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ACTION_AGENT_AZURE_API_KEY = os.environ.get("ACTION_AGENT_AZURE_API_KEY")

class ContextAgent:
    def __init__(self, tokenizer) -> None:
            
        self.client = AsyncAzureOpenAI(
            api_key=ACTION_AGENT_AZURE_API_KEY,
            api_version="2023-10-01-preview",
            azure_endpoint = "https://action-agents.openai.azure.com/"
            )

        self.config_list = [
            {
                'model': "gpt-4-1106-preview",
                'api_key': OPENAI_API_KEY,
            },
        ]

        self.system_message = CONTEXT_PROMPT

        self.config_list_copy = self.config_list.copy()
        self.config_list_copy[0]["response_format"] = {"type": "json_object"}

        self.llm_config = {
            "config_list": self.config_list_copy,
            "timeout": 120,
            # "cache_seed": None
        }
        self.tokenizer = tokenizer

    async def execute(self,user_query:str,chat_history:str)-> List[dict]:
        try:
            data_processor = DataProcessing(self.tokenizer)
            results = data_processor.query_results(user_query)

            self.system_message += f"\nChat history:\n{chat_history}\nContext:\n{results}"
        
            response = await self.client.chat.completions.create(
                stream=False,
                model="agent-api",
                messages=[{"role": "system", "content": self.system_message}],
            )
            output_message = response.choices[0].message.content
            print("OUTPUT CONTEXT AGENT", output_message)

            formatted_message = output_message.split("AI:")[-1]
            return formatted_message 
        except Exception as e:
            print("ERROR at context agent execution", e )