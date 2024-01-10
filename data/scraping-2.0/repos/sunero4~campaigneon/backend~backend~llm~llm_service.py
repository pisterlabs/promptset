import json
import os
from typing import List
from backend.db.models import Entry
import requests
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class GenerateImageResponse:
    def __init__(self, created_at: int, url: str) -> None:
        self.created_at = created_at
        self.url = url

GENERATE_DESCRIPTION_PROMPT = """
From the text below, generate a short, concise summary focusing on descriptions of visuals. Include any relevant adjectives describing the appearance of persons or scenery in the text.
Do not include the word '{entry_title}' in the summary, and also try to avoid including any words that might be names of persons or locations.

Here are a few examples of prompts, and good responses to those:

EXAMPLE 1:
PROMPT: Stronkus is the capital city of Eon. It is a large medieval city surrounded by thick stone walls to keep out any intruders. At the center of Stronkus is a large hill. At the top of the hill is an enormous fortress where the king resides.
Outside the city walls there is forest to all sides, as far as the eye can see. Beyond the forest lies the vast Vanyon mountains.
RESPONSE: Large medieval city, stone walls, enormous fortress at the top of hill, surrounded by forest, mountains in background

EXAMPLE 2:

PROMPT: The village of Heimer was established by the elves many hundred years ago. It is built high up in the trees of the jungle, to keep the elves safe from the many dangerous animals that roam the grounds below.
Their huts balance on platforms attached to the trunks of the large trees. The village is very dark, as the foliage of the surrounding trees keep out most of the sunlight.
RESPONSE: Elven village, huts on platforms in treetops, jungle, dark

EXAMPLE 3:
PROMPT: Aeldun is the god of lightning. Together with the two other weather gods, Wearn and Harkes, he controls the weather of the world.
He is usually depicted as a large man with purple skin and a huge beard, floating atop a dark cloud with lightning coming out of it. He wields a trident with lightning-shaped prongs.
RESPONSE: Lightning god with purple skin, huge beard, floats on storm cloud, wields trident

PROMPT:
{description}

YOUR RESPONSE:
"""

class LlmService:
    def __init__(self) -> None:
        self.__api_key = os.environ.get("OPENAI_API_KEY")

    def generate_image_for_entry(self, entry: Entry, art_styles: List[str]) -> GenerateImageResponse:
        image_generation_prompt = self.generate_entry_image_generation_prompt(entry=entry, art_styles=art_styles)

        print(image_generation_prompt)

        return self.generate_image(image_generation_prompt)
        
    def generate_entry_image_generation_prompt(self, entry: Entry, art_styles: List[str]) -> str:
        llm = OpenAI(temperature=0.7, api_key=self.__api_key)

        prompt_template = PromptTemplate(
            template=GENERATE_DESCRIPTION_PROMPT,
            input_variables=["entry_title", "description"]
        )

        chain = LLMChain(llm=llm, prompt=prompt_template)

        result = chain.run(entry_title=entry.title, description=entry.entry_text_raw)
        art_styles_str = " ".join(art_styles)

        return f"{result} | {art_styles_str}"

    def generate_image(self, prompt: str) -> GenerateImageResponse:
        request_body = json.dumps(
            {
                "model": "dall-e-2",
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
            }
        )

        response = requests.post(
            url="https://api.openai.com/v1/images/generations",
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.__api_key}",
            },
        )

        if response.status_code < 200 or response.status_code > 299:
            if response.status_code == 429:
                raise Exception("Rate limit exceeded.")
            else:
                raise Exception("Something went wrong while trying to generate image.")

        response_content = response.json()

        return GenerateImageResponse(
            created_at=response_content["created"],
            url=response_content["data"][0]["url"],
        )
