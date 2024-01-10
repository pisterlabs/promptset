import os
import dotenv
import json
from fuzzywuzzy import fuzz
from typing import List
from openai import AsyncAzureOpenAI
from timeit import default_timer as timer
from src.logging.cache import recommended_prod_cache
from src.shopping_flow.agents.constants import SECONDARY_EXCLUDED_FIELDS

dotenv.load_dotenv()
ACTION_AGENT_AZURE_API_KEY = os.environ.get("ACTION_AGENT_AZURE_API_KEY")

class ProductAgentAssistant:
    def __init__(self) -> None:
        self.client = AsyncAzureOpenAI(
            api_key=ACTION_AGENT_AZURE_API_KEY,
            api_version="2023-10-01-preview",
            azure_endpoint = "https://action-agents.openai.azure.com/"
            )
    
    async def execute(self, user_query: str, chat_history: str) -> List[str] | str:
        try:
            print("entering paa")
            start = timer()
            
            ASSISTANT_PROMPT = """
            Given a chat history between AI and User, determine if the User's current query is referring to or is a follow up to a product/products previously recommended by the AI. If it is not referring to a previous product, output empty string. If it is, extract the product's name(s) to the best of your ability. If user uses a singular pronoun, default to first product in most recent recommendation. If unclear default to empty string.
            Output in strictly list of JSON format as described below:
            \{"output": empty string/[list of product name(s)]}
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
            print("PAA TIME", end - start)
            
            return output_json["output"]
        except Exception as e:
            print("ERROR AT PAA EXECUTION", e)
        
    async def answer_query(self, user_id, user_query: str, mentioned_products: List[str]) -> str:
        try:
            start = timer()
            product_list = await self.find_canonical_product_name(mentioned_products, list(recommended_prod_cache[user_id].keys()))
            product_details_list = [v for k,v in recommended_prod_cache[user_id].items() if k in set(product_list)]
            
            truncated_product_list = await self.truncate_product_details(product_details_list)
            ASSISTANT_PROMPT = """
            You're a helpful AI copilot @ Miller Family Wine Company. Given a list of product details and user query, use the product details to answer the user's query. Push user down sales path but remember to be friendly and helpful. Don't reference any URL's. Be concise and respond in no more than 2 sentences. It's ok if you don't know the answer to the user's query, just acknowledge that in your response and don't make anything up. Output in strictly list of JSON format as described below:
            \{"output": answer to user query}
            """
            ASSISTANT_PROMPT += f"\nUser query: {user_query}\nProduct details: {truncated_product_list}"
            
            stream =  await self.client.chat.completions.create(
                model="agent-api",
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": ASSISTANT_PROMPT}],
            )
            output_message = stream.choices[0].message.content
            output_json = json.loads(output_message)
            end = timer()
            print("paa answering time", end - start)
            return output_json["output"], product_details_list
        except Exception as e:
            print("ERROR at paa answering query", e)
            
    async def find_canonical_product_name(self, input_strings: List[str], target_strings: List[str]) -> List[str]:
        start = timer()
        results = []

        for input_string in input_strings:
            print("curr input", input_string)
            highest_similarity = 0
            most_similar_string = None

            for target_string in target_strings:
                similarity = fuzz.ratio(input_string, target_string)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_string = target_string

            results.append(most_similar_string)
        
        end = timer()
        print("canonical product name time", end - start)
        return results
    
    async def truncate_product_details(self, product_details):
        truncated_product_details = []
        for (index, product_dic) in enumerate(product_details):
            new_product_dic = {k: v for k, v in product_dic.items() if (k not in set(SECONDARY_EXCLUDED_FIELDS) and v is not None)}
            truncated_product_details.append(new_product_dic)
        return truncated_product_details