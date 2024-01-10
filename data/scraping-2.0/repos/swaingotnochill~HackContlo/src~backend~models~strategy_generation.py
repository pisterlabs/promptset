import json
import openai
openai.api_key = "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"

class StrategyGenerator:
    def __init__(self, prompt_file:str="src/backend/configs/prompts.json") -> None:
        with open(prompt_file) as f:
            self.prompts = json.load(f)

    def get_retention_strategy(self, user_persona:str, segmention_type:str) -> str:
        retention_message = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {
                "role": "system", 
                "content":self.prompts["strategy"]["base_prompt"]
            },
            {
                "role": "user",
                "content": self.prompts["strategy"]["retention_prompt"].format(
                                    user_persona=user_persona,
                                    segmention_type=segmention_type)
            }
            ]
        )
        return retention_message["choices"][0]["message"]["content"]

    def get_aquisition_strategy(self, segmention_type:str, product:str) -> str:
        aquisition_message = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {
                "role": "system", 
                "content":self.prompts["strategy"]["base_prompt"]
            },
            {
                "role": "user",
                "content": self.prompts["strategy"]["aquisition_prompt"].format(
                                    segmention_type=segmention_type,
                                    product=product)
            }
            ]
        )
        return aquisition_message["choices"][0]["message"]["content"]
