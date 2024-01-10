import os
import dotenv
import json
from fuzzywuzzy import fuzz
from typing import List, Optional, Dict, Tuple
from openai import AsyncAzureOpenAI
from timeit import default_timer as timer
from src.shopping_flow.agents.constants import AVAILABLE_WINE_TYPES, SECONDARY_EXCLUDED_FIELDS, PRIMARY_EXCLUDED_FIELDS
from src.logging.cache import more_suggestions_cache, recommended_prod_cache, recommended_product_links

dotenv.load_dotenv()
ACTION_AGENT_AZURE_API_KEY = os.environ.get("ACTION_AGENT_AZURE_API_KEY")

class AlternativesProductAgent:
    def __init__(self, user_id: str) -> None:
        self.client = AsyncAzureOpenAI(
            api_key=ACTION_AGENT_AZURE_API_KEY,
            api_version="2023-10-01-preview",
            azure_endpoint = "https://action-agents.openai.azure.com/"
            )
        self.user_id = user_id
    
    async def execute(self, user_query: str, chat_history: str) -> str:
        try:
            start = timer()
            alternative_wine_types = await self.get_wines_for_alternatives(user_query, chat_history)
            alternative_details = await self.get_alternative_details(alternative_wine_types)
            response = await self.get_ai_response(alternative_details, user_query)
            end = timer()
            return response
        except Exception as e:
            print("ERROR at APA execution", e)
        
    async def get_wines_for_alternatives(self, user_query: str, chat_history: str) -> List[str] | str:
        try:
            start = timer()
            
            ASSISTANT_PROMPT = f"""
            You're given chat history b/w AI and User as well as User's most recent user query. You're an AI copilot @ Miller Family Wine Company. You've just suggeted some wines, but the User wants some alternatives. Come up with a list of wine types if applicable that the user has expressed wanting alternatives for. Here are the available wine types: 
            {AVAILABLE_WINE_TYPES}
            Output in strictly list of JSON format as described below:
            {{"output": [list of wines]}}
        
            """
            ASSISTANT_PROMPT += f"\nUser query: {user_query}\nChat history: {chat_history}"
            stream =  await self.client.chat.completions.create(
                        model="agent-api",
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": ASSISTANT_PROMPT}],
                    )
            
            output_message = stream.choices[0].message.content
            output_json = json.loads(output_message)
            
            end = timer()
            print("alternatives times", end - start)
            
            return output_json["output"]
        except Exception as e:
            print(e)
    
    async def get_alternative_details(self, wine_types: List[str]) -> Dict[str, Optional[List[Dict]]]:
        # Cache dictionaries
        try:
            alternatives = more_suggestions_cache.get(self.user_id, {})
            curr_recs = recommended_prod_cache.get(self.user_id, {})
            curr_links = recommended_product_links.get(self.user_id, {})

            # Output result dictionary
            result = {}
            total_alternatives = 0

            # Iterate to add alternatives, prioritizing diversity in wine types
            for _ in range(3):  # Loop up to 3 times to potentially add 3 alternatives
                for wine_type in wine_types:
                    if total_alternatives >= 3:
                        break  # Stop if 3 alternatives have been added

                    if wine_type in alternatives and alternatives[wine_type]:
                        if wine_type not in result or len(result[wine_type]) < 3:
                            product_detail = alternatives[wine_type].pop(0)
                            result.setdefault(wine_type, []).append(product_detail)
                            curr_recs[product_detail['productName']] = product_detail
                            curr_links[product_detail['productName']] = product_detail['productUrl']
                            total_alternatives += 1

                            # If the list for this wine type is empty, remove the key
                            if not alternatives[wine_type]:
                                del alternatives[wine_type]
                    elif wine_type not in result:
                        # Wine type not available in alternatives
                        result[wine_type] = False

                    # Check if we have exhausted all alternatives
                    if not any(alternatives.values()):
                        break

                # Exit the outer loop if we have exhausted all alternatives
                if not any(alternatives.values()):
                    break

            # Update the caches
            more_suggestions_cache[self.user_id] = alternatives
            recommended_prod_cache[self.user_id] = curr_recs
            recommended_product_links[self.user_id] = curr_links

            return result
        except Exception as e:
            print("ERROR AT apa get alternative details", e)

    async def get_ai_response(self, alternative_details: Dict[dict, any], user_query: str) -> Tuple[str, List[dict]]:
        try:
            start = timer()
            
            truncated_details = await self.truncate_product_details(alternative_details)
            products = await self.extract_products(alternative_details)
            
            ASSISTANT_PROMPT = f"""
            You're given the User's most recent user query. You're an AI copilot @ Miller Family Wine Company. You've just suggeted some wines, but the User wants some alternatives. You're also given the product details for the alternatives. Push user down sales path but remember to be friendly and helpful. Reference as many wines as possible in your response. Be concise!! Response no longer than 2 sentences and be concise. Try to mention three products, but only dive into the first. If any of the wines are award winning, mention that. Do not reference any links. Avoid lists and use natural language formatting. If the value is "False" for any of the alternatives, that means you should ask the user for more details for that wine type. Please do not make up any wine details, your response should be based off of the details provided. Output in strictly list of JSON format as described below:
            {{"output": your response}}
            
            User Query: {user_query}
            Alternative Product details: {truncated_details}
            """
            
            stream =  await self.client.chat.completions.create(
                        model="agent-api",
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": ASSISTANT_PROMPT}],
                    )
            
            output_message = stream.choices[0].message.content
            output_json = json.loads(output_message)
            
            end = timer()
            print("alternatives times", end - start)
            
            return output_json["output"], products
        except Exception as e:
            print(e)
    
    async def extract_products(self, alternative_wines):
        products = []
        for wine_type, product_details in alternative_wines.items():
            if product_details == False:
                continue
            for product in product_details:
                products.append(product)
        
        return products
    
    async def truncate_product_details(self, alternative_wines):
        truncated_product_details = {}
        i = 0
        for wine_type, product_details in alternative_wines.items():
            truncated_details = []
            if product_details == False:
                truncated_product_details[wine_type] = product_details
                continue
            if i == 0:
                for product in product_details:
                    new_product_dic = {k: v for k, v in product.items() if (k not in set(PRIMARY_EXCLUDED_FIELDS) and v is not None)}
                    truncated_details.append(new_product_dic)
            else:
                for product in product_details:
                    new_product_dic = {k: v for k, v in product.items() if (k not in set(SECONDARY_EXCLUDED_FIELDS) and v is not None)}
                    truncated_details.append(new_product_dic)
            i += 1
            truncated_product_details[wine_type] = truncated_details
            
        return truncated_product_details
    
    
    