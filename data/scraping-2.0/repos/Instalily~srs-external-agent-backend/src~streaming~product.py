import os
import autogen
import dotenv
import json
from openai import OpenAI
from typing import List, Optional, Tuple, Dict
from src.autogen_modules.orchestrator import Orchestrator
from src.autogen_modules.async_orchestrator import AsyncOrchestrator
from src.shopping_flow.vespa_app import VespaApp
from src.shopping_flow.traits_extractor import WineTraits, OtherTraits
from pydantic import BaseModel, Field, RootModel
from src.shopping_flow.agents.constants import DEFAULT_WINE_DF, DEFAULT_OTHER_DF, PRODUCT_AGENT_PROMPT_MAPPING

dotenv.load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class ProductStreamer:
    def __init__(self) -> None:
       self.client = OpenAI()

    def need_followup(self, product_traits: Dict[str, any]) -> bool:
        """
        Determines if a follow-up is required based on the extracted product traits.

        Returns:
            bool: True if follow-up is needed, False otherwise.
        """
        # No follow-up if SKU or Product Name is available
        if product_traits.get('sku') or product_traits.get('productName'):
            return False

        # Count non-null traits excluding SKU and Product Name
        non_null_traits = sum(1 for key, value in product_traits.items() 
                              if value and key not in ['sku', 'productName'])

        # Follow-up is needed if no non-null traits
        if not non_null_traits:
            return True

        return False

    # TODO:
    # handle vespa errors, cases where results are None, invalid YQL etc,
    async def findProduct(self, chatHistory: str, productType: str) -> Tuple[List[dict], bool]:
        """
        Returns: results (dict), needs_followup question (bool)
        """
        if productType == "wine":
            product_traits = await WineTraits().extract_traits(chatHistory)
            product_traits = product_traits.to_dict()
        else:
            product_traits = await OtherTraits().extract_traits(chatHistory)
            product_traits = product_traits.to_dict()
        
        # now we should count up the product_traits, and determine if we need to raise a follow up question
        if self.need_followup(product_traits):
            return {}, True
        try:
            product_recommender = VespaApp()
            recommended_product = product_recommender.get_product_rec(productType, product_traits)
            results = recommended_product.to_dict("records")
            self.results = results
            return results, False
        except Exception as e: #TODO: handle and raise vespa error?
            return {}, False

    def handle_no_results(self, product_type: str) -> List[dict]:
        if product_type == "wine": 
            self.no_results = DEFAULT_WINE_DF
            return DEFAULT_WINE_DF
        self.no_results = DEFAULT_OTHER_DF
        return DEFAULT_OTHER_DF
    
    async def execute(self, chat_history: str, product_type: str) -> List[dict]:
        from timeit import default_timer as timer
        start = timer()
        maybe_results, need_followup = await self.findProduct(chat_history, product_type)
        results = maybe_results
        if not maybe_results:
            results = self.handle_no_results(product_type)
        json_output = await self.execute_agent(chat_history, results, maybe_results, need_followup)
        end = timer()
        print(" EXECUTE TIME", end - start)
        return json_output
    
    def truncate_product_details(self, product_details: List[dict]) -> List[dict]:
        truncated_product_details = []
        for product_dic in product_details:
            new_product_dic = {k: v for k, v in product_dic.items() if (k not in ['sku', 'year', 'imageUrl', 'productUrl', 'type', 'color'] and v is not None)}
            truncated_product_details.append(new_product_dic)
        return truncated_product_details
    
    async def execute_agent(self, chat_history: str, product_details: List[dict], vespa_results: bool, need_followup: bool) -> List[dict]:
        had_vespa_results = True if vespa_results else False
        
        PRODUCT_AGENT_PROMPT = PRODUCT_AGENT_PROMPT_MAPPING[(had_vespa_results, need_followup)]
        
        truncated_product_details = self.truncate_product_details(product_details)
        
        product_prompt = f"Chat history:\n{chat_history}\nProduct details:\n{truncated_product_details}"
        
        PRODUCT_AGENT_PROMPT += product_prompt
        
        await self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            stream=True,
            messages=[{"role": "system", "content": PRODUCT_AGENT_PROMPT}],
        )

        try:
            json_output = json.loads(output_message)
            print("json output", json_output)
            json_output["products"] = product_details
            return json_output
        except Exception as e:
            try:
                parsed_data = RootAgentResponse.model_validate_json(run_output.messages[-1])
                agent_sequences = parsed_data.root
                sequences_dict = [dict(item) for item in agent_sequences]
                sequences_dict[0]["products"] = product_details
                return sequences_dict
            except Exception as exc:
                print('JSON parsing during fn execute_agent file product_agent.py failed:', e)
                print('Pydantic parsing during fn execute_agent file product_agent.py failed:', exc)
                return None
            
    