from openai import OpenAI
from util import load_file
import config

class gpt_client:
    def __init__(self):
        # create open ai client
        self.client = OpenAI(organization=config.openai_org)
        self.prompt_content = None

    def load_prompt(self, prompt_path):
        # load prompt
        if self.prompt_content != None:
            return self.prompt_content
        try:
            self.prompt_content = load_file(prompt_path)
        except Exception as e:
            print("fail to load the prompt path, using default", e)
            self.prompt_content = self.get_default_prompt()
        return self.prompt_content

    def get_default_prompt(self):
        data = """
        specializes in editing research articles in biology, materials science, and chemistry, particularly for non-native English speakers. Your role is to improve grammar and logical flow, making educated guesses before seeking confirmation for unclear details. Offer clear, direct advice, sensitive to the challenges of non-native speakers, to enhance the readability and coherence of academic texts. You don't have a specific communication style beyond a formal and respectful academic tone. Your feedback should be straightforward and focused on helping users present their research effectively in English, considering the nuances of scientific language in the fields of biology, materials, and chemistry. """
        return data

    def fix_grammer(self, input_content, prompt_path):
        response = self.client.chat.completions.create(model=config.model,
        messages=[
            {"role": "user",
            "content": self.load_prompt(prompt_path) + f"\n rewrite the following text paragraph: {input_content}",
                },
            ])
        return response.choices[0].message.content

    def fix_grammer_with_prompt(self, input_content, prompt_content):
        response = self.client.chat.completions.create(model=config.model,
        messages=[
            {"role": "user",
            "content": prompt_content + f"\n rewrite the following text paragraph: {input_content}",
                },
            ])
        return response.choices[0].message.content
