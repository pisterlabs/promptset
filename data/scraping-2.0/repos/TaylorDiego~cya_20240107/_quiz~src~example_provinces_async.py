
import time
import asyncio
import aiohttp
import json
import yaml


from openai import OpenAI


class ProvinceFacts():
    """Get facts about Canadian provinces"""
    def __init__(self, config_path=r'..\config\config.yaml'):
        """Initialize the ProvinceFacts class"""
        self.provinces = ["Ontario", "British Columbia"] #, "Alberta", "Quebec", "Manitoba", "Saskatchewan", "Nova Scotia", "New Brunswick", "Newfoundland and Labrador", "Prince Edward Island", "Northwest Territories", "Nunavut", "Yukon"]
        self.province_facts = {}
        self.CONFIG = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
        self.CONTEXT = self.CONFIG["CONTEXT"]
        self.TASK = self.CONFIG["TASK"]
        self.TASK_FORMAT = self.CONFIG["TASK_FORMAT"]

        self.KEYS = yaml.safe_load(open(self.CONFIG["KEYS"], 'r', encoding='utf-8'))
        self.OPENAI_API_KEY = self.KEYS["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)

    async def province_year(self, client, province_name):
        """Get the year a province was founded"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo"
            , temperature = 0
            , max_tokens = 200
            , messages = [
                {"role": "system", "content": "You are an expert in Canadian history."}
                , {"role": "user", "content": f"What year was {province_name} named the current name (as of last information cut-off date)?"}
                , {"role": "system", "content": "Please keep your answer short and to the point."}
            ]
        )
        result = await response.choices[0].message.content
        # print(result)
        return await result

    async def run_facts(self):
        """Run the facts"""
        async with aiohttp.ClientSession() as session:
            for province in self.provinces:
                print(f"\nGetting facts for: {province}")
                self.province_facts[province] = await self.province_year(self.client, province)
        return await self.province_facts

    def start_user(self):
        """Start the user"""
        print("Let's learn about Canadian history!\n")

    async def main(self):
        """Run the main program"""
        self.start_user()
        facts = await self.run_facts()
        print(f"\nHere are the facts about Provinces: \n{facts}")

if __name__ == "__main__":
    province_facts = ProvinceFacts()
    asyncio.run(province_facts.main())