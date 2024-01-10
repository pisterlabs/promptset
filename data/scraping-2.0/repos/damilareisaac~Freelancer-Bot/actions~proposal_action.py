from functools import partial
import os
import json
import openai
from typing import Any

from logs import get_logger

logger = partial(get_logger, __name__)
default_proposal = """I have read through your project description,\
I can help you complete the project. I will be looking forward to hearing from you \
to discuss it further."""


class ProposalAction:
    def __init__(self) -> None:
        open_ai_org_key: str = str(os.environ.get("OPEN_AI_ORG_KEY"))
        open_ai_api_key: str = str(os.environ.get("OPEN_AI_API_KEY"))
        openai.organization = open_ai_org_key
        openai.api_key = open_ai_api_key
        self.bid_cache_file_path = "bid_cache.json"

    def get_proposal(self, description) -> str:
        hint = str(
            f"""
            I can help you complete your  project stated as follows 
            {description}

            """
        )
        try:
            res: Any = openai.Completion.create(
                model="text-davinci-003",
                prompt=hint,
                max_tokens=200,
                temperature=0.5,
            )
            response_text = res.choices[0].text.strip()
            if not response_text:
                logger(to_console=True).error("Empty proposal from OPENAI")
                return default_proposal
            if len(response_text)  > 1500:
                logger().info(f"OPENAI response of lenght {len(response_text)} is too long, truncating...")
                return f"{response_text[:1400]}..."
                
            return response_text
        except Exception as e:
            logger(to_console=True).exception(f"Error in OPENAI: {e}")
            return default_proposal
        
    def read_bid_cache(self):
        with open("bid_cache.json", "r+") as f:
            try:
                json_file = json.load(f)
            except Exception as e:
                logger().exception(f"Error reading bid json file: {e}")
                json_file = {}
        return json_file


    def update_bid_cache(self, link, proposal):
        bid_cache = self.read_bid_cache()
        with open(self.bid_cache_file_path, "w+") as f:
            bid_cache.update({link:proposal})
            json.dump(bid_cache, f, indent=4)

    def delete_proposal_from_cache(self, link):
        bid_cache = self.read_bid_cache()
        if link in bid_cache:
            del bid_cache[link]
        with open(self.bid_cache_file_path, "w+") as f:
            json.dump(bid_cache, f, indent=4)

    
    def check_proposal_in_cache_for_link(self, link):
       bid_cache = self.read_bid_cache()
       if link in bid_cache:
           return bid_cache["link"]
       return None
