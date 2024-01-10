
import openai
from lib.gpt3.gpt3_api import GPT3API

class MarketingAutomator:
    def __init__(self):
        self.gpt3_api = GPT3API()

    def generate_ad(self, product):
        prompt = f"Create an engaging ad for a product: {product}"
        response = self.gpt3_api.generate_text(prompt)
        return response

    def generate_social_media_post(self, product):
        prompt = f"Create a social media post promoting the product: {product}"
        response = self.gpt3_api.generate_text(prompt)
        return response

    def generate_email_campaign(self, product):
        prompt = f"Create an email campaign for the product: {product}"
        response = self.gpt3_api.generate_text(prompt)
        return response
