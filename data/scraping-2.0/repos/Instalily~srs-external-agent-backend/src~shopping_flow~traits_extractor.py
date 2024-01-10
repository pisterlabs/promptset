import logging as log
from pydantic import BaseModel
from typing import List, Optional, Dict
import openai
from openai import OpenAI, AsyncOpenAI
from timeit import default_timer as timer
import instructor
from dotenv import load_dotenv
from src.shopping_flow.prompts import OTHER_TRAITS_EXTRACTION_PROMPT, WINE_TRAITS_EXTRACTION_PROMPT
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

client = instructor.patch(AsyncOpenAI())
class WineTraits(BaseModel):
    productName: Optional[str] = None
    price: Optional[dict] = None
    year: Optional[int] = None
    vendor: Optional[str] = None
    type: Optional[str | List[str]] = None
    color: Optional[str] = None
    alcoholic: Optional[bool] = True
    award_winning: Optional[bool] = None

    def to_dict(self):
        return {
            'productName': self.productName,
            'price': self.price,
            'year': self.year,
            'vendor': self.vendor,
            'type': self.type,
            'color': self.color,
            'alcoholic': self.alcoholic,
            'awardWinning': self.award_winning,
        }
    
    def __str__(self):
        return self.model_dump_json()

    def prompt(self, chat_history: List[str]) -> str:
        wine_extraction_prompt = WINE_TRAITS_EXTRACTION_PROMPT + f"\nChat history:\n{chat_history}"
        return wine_extraction_prompt

    async def extract_traits(self, chat_history:List[str]):
        start = timer()
        extracted_traits: WineTraits = await client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_model=WineTraits,
            max_retries=5,
            messages=[{"role": "system", "content": self.prompt(chat_history)}],
        )
        end = timer()
        print("EXTRACTION TIME", end - start)
        return extracted_traits
    
# Extraction class for products that aren't wines    
class OtherTraits(BaseModel):
    productName: Optional[str] = None
    sku: Optional[str] = None
    price: Optional[float] = None
    vendor: Optional[str] = None
    other: Optional[str] = None

    def __str__(self):
        return self.model_dump_json()

    def to_dict(self):
        return {
                'productName': self.productName,
                'sku': self.sku,
                'price': self.price,
                'vendor': self.vendor,
                'other': self.other,
            }

    def prompt(self, chat_history: List[str]) -> str:
        other_extraction_prompt = OTHER_TRAITS_EXTRACTION_PROMPT + f"\nChat history:\n{chat_history}"
        return other_extraction_prompt

    def extract_traits(self, chat_history:List[str]):
        start = timer()
        extracted_traits: OtherTraits =  client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_model=OtherTraits,
            max_retries=5,
            messages=[{"role": "system", "content": self.prompt(chat_history)}],
        )
        end = timer()
        print("EXTRACTION TIME", end - start)
        return extracted_traits
