import os
import dotenv
import json
from typing import List, Optional, Tuple, Dict
from openai import AsyncAzureOpenAI
from src.shopping_flow.vespa_app import VespaApp
from pydantic import BaseModel, Field, RootModel
from src.shopping_flow.agents.constants import DEFAULT_WINE_DF, DEFAULT_OTHER_DF, AVAILABLE_WINE_TYPES, SECONDARY_EXCLUDED_FIELDS, PRIMARY_EXCLUDED_FIELDS
from src.shopping_flow.agents.prompts import PRODUCT_AGENT_PROMPT_MAPPING, WINE_TYPE_REC_PROMPT
import src.logging.cache as logging
from timeit import default_timer as timer

dotenv.load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ACTION_AGENT_AZURE_API_KEY = os.environ.get("ACTION_AGENT_AZURE_API_KEY")
class AgentResponse(BaseModel): 
    ai_chat_response: str = Field(description="Response from the AI referencing at most two products")
    is_recommendation: bool = Field(description="Is this a product recommendation?")
    products: Optional[List[dict]] = Field(default=[], description="Complete output from findProduct function as is including all products found")
class RootAgentResponse(RootModel):
    root: List[AgentResponse] = Field(description="Sequence of agent responses")
class ProductAgent:
    def __init__(self, user_uuid) -> None:
        self.user_uuid = user_uuid
        self.client = AsyncAzureOpenAI(
            api_key=ACTION_AGENT_AZURE_API_KEY,
            api_version="2023-10-01-preview",
            azure_endpoint = "https://action-agents.openai.azure.com/"
            )
        self.results = None
        
    async def refine_traits(self, wine_traits: Dict[str, any], chat_history: str) -> Dict[str, any]:
        try:
            valid_wine_type = set(AVAILABLE_WINE_TYPES)
            if wine_traits.get("alcoholic") == False or isinstance(wine_traits.get("type"), list) or wine_traits.get("type") in valid_wine_type or wine_traits.get("color") == "rose":
                return wine_traits
            
            recommended_wine_types = await self.give_wine_type_recommendation(chat_history)
            if not recommended_wine_types:
                recommended_wine_types = ["chardonnay", "pinot noir"]
            
            wine_traits["type"] = recommended_wine_types
            return wine_traits
        except Exception as e:
            print("ERROR at refining traits", e)
    

    async def findProduct(self, productType: str, product_traits: Dict[str, any], chat_history: str) -> Tuple[List[dict], bool]:
        """
        Determines whether or not a followup question is needed. If not, 
        given the extracted product traits and the type of product ("wine"/"other"), it will conduct the Vespa search and returns the results as a list of dictionaries. Each dictionary corresponds to the details of a unique product. 
        
        Returns: 
            The list of product details, whether or not a followup is needed
        """
        # tier 1 --> nonalcoholic, price, color, type
        # tier 2 --> award winning, year, vendor
        had_to_retry_vespa = False
        try:
            start = timer()
            refined_product_traits = product_traits
            product_recommender = VespaApp()
            productType = "wine"
            if productType == "wine":
                refined_product_traits = await self.refine_traits(product_traits,chat_history)
                print("PRODUCT TRAITS", product_traits)
                print("REFINED PRODUCT TRAITS", refined_product_traits)
                
                if isinstance(refined_product_traits, list) and len(refined_product_traits):
                    recommended_product = product_recommender.get_product_rec_multiple_types(refined_product_traits)
                    
                else:
                    recommended_product =  product_recommender.get_product_rec(productType, refined_product_traits)
                    
                # TODO: include in the caching s33ystem???
                alternatives = product_recommender.get_alternative_top_results(refined_product_traits)
                
            else:
                recommended_product =  product_recommender.get_product_rec(productType, refined_product_traits)
            
            results = recommended_product.to_dict("records")
            self.results = results
            if product_recommender.had_to_retry: had_to_retry_vespa = True
            end = timer()
            await self.add_products_to_user_cache(results, alternatives)
            return results, had_to_retry_vespa
        except Exception as e: #TODO: handle and raise vespa error?
            print("PRODUCT AGENT ERROR", e)
            return [], had_to_retry_vespa

    async def add_products_to_user_cache(self, recommended_prods: List[dict], alternatives: Dict[str, List[Dict[str, any]]]) -> None:
        try:
            if not self.user_uuid: 
                return
            
            logging.more_suggestions_cache[self.user_uuid] = alternatives
            
            # Get the current recommendations and links, or initialize them as empty dicts
            curr_recommendations = logging.recommended_prod_cache.get(self.user_uuid, {})
            curr_links = logging.recommended_product_links.get(self.user_uuid, {})
            
            for prod in recommended_prods:
                prod_dic = {prod["productName"]: prod}
                prod_link = {prod["productName"]: prod["productUrl"]}

                # Update the dictionaries without reassigning
                curr_recommendations.update(prod_dic)
                curr_links.update(prod_link)

            # Now assign the updated dictionaries back to the logging module
            logging.recommended_prod_cache[self.user_uuid] = curr_recommendations
            logging.recommended_product_links[self.user_uuid] = curr_links
        except Exception as e:
            print("ERROR at adding products to user cache", e)

            
    def handle_no_results(self, product_type: str) -> List[dict]:
        """
        If no results are returned in the vespa search, we set a class field to either the default wine recs or default other product recs.
        """
        if product_type == "wine": 
            self.no_results = DEFAULT_WINE_DF
            return DEFAULT_WINE_DF
        self.no_results = DEFAULT_OTHER_DF
        return DEFAULT_OTHER_DF
    
    async def execute(self, chat_history: str, product_type: str, product_traits: Dict[str, any], need_ai_response: bool) -> List[dict]:
        """
        Entry point for product agent.
        """
        try:
            start = timer()
            
            maybe_results, retried_vespa =  await self.findProduct(
                product_type, 
                product_traits,
                chat_history
            )
            results = maybe_results
            
            # Assign default recs if we didn't get any vespa results
            if not maybe_results: 
                results = self.handle_no_results(product_type)
                
            had_vespa_results = True if len(maybe_results) else False
            
            json_output = await self.execute_agent(
                chat_history, 
                results, 
                had_vespa_results, 
                retried_vespa,
                need_ai_response
            )
            
            end = timer()
            return json_output
        except Exception as e:
            print("ERROR", e)
    
    def truncate_product_details(self, product_details: List[dict]) -> List[dict]:
        """
        Given a list of 1 - 3 product detail dictionaries, returns a new list of product detail dictionaries with fewer k/v pairs.
        """
        truncated_product_details = []
        for (index, product_dic) in enumerate(product_details):
            if index > 0:
                new_product_dic = {
                    k: v for k, v in product_dic.items() if (k not in set(PRIMARY_EXCLUDED_FIELDS) and v is not None)
                    }
            else:
                new_product_dic = {
                    k: v for k, v in product_dic.items() if (k not in set(SECONDARY_EXCLUDED_FIELDS) and v is not None)
                    }  
            truncated_product_details.append(new_product_dic)
            
        return truncated_product_details
    
    async def give_wine_type_recommendation(self, chat_history: str, previous_recs: Optional[list]=None):
        try:
            wine_type_prompt = WINE_TYPE_REC_PROMPT
            wine_type_prompt += f"\nChat history: {chat_history}"
            if previous_recs: 
                wine_type_prompt += f"\nPreviously recommended: {previous_recs}"
            
            stream = await self.client.chat.completions.create(
                    model="agent-api",
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": wine_type_prompt}],
                )
            output_message = stream.choices[0].message.content
            output_json = json.loads(output_message)
            wine_recommendations = output_json["output"]
            print("wine type prompt", wine_type_prompt)
            print("output", output_json)
            print("WINE RECOMMENDATIONS", wine_recommendations)
            return wine_recommendations
        except Exception as e:
            print("ERROR", e)

    async def execute_agent(self, chat_history: str, product_details: List[dict], vespa_results: bool, had_to_retry: bool, need_ai_response: bool) -> List[dict]:
        """
        Execution function for the product agent.
        """
        try:
            json_output = {}
            
            # If we need an ai response, we must call the LLM to synthesize the product recommendation into natural language.
            if need_ai_response:
                start = timer()
                
                # Get the appropriate prompt
                PRODUCT_AGENT_PROMPT = PRODUCT_AGENT_PROMPT_MAPPING[(vespa_results, had_to_retry)]
                
                # Truncate product details
                truncated_product_details = self.truncate_product_details(product_details)
                            
                PRODUCT_AGENT_PROMPT += f"\nChat history:\n{chat_history}\nProduct details:\n{truncated_product_details}"

                stream = await self.client.chat.completions.create(
                    model="agent-api",
                    stream=False,
                    messages=[{"role": "system", "content": PRODUCT_AGENT_PROMPT}],
                )
                output_message = stream.choices[0].message.content
                
                end = timer()
                
                json_output["ai_chat_response"] = output_message
            
            # Regardless, always assign products to the list of product details dictionaries and "is_recommendation" to True
            json_output["products"] = product_details
            json_output["is_recommendation"] = True
            return json_output
        except Exception as e:
            try:
                parsed_data = RootAgentResponse.model_validate_json(output_message)
                agent_sequences = parsed_data.root
                sequences_dict = [dict(item) for item in agent_sequences]
                sequences_dict[0]["products"] = product_details
                return sequences_dict
            except Exception as exc:
                print('JSON parsing during fn execute_agent file product_agent.py failed:', e)
                print('Pydantic parsing during fn execute_agent file product_agent.py failed:', exc)
                return None
            
    