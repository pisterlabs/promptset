import openai
import os
from dotenv import load_dotenv

# Generate the api key from https://platform.openai.com/account/api-keys
ENV_PATH = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path=ENV_PATH)

class ExtractSummaryGPT():
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-3.5-turbo"
        self.role = "system"
        self.prompt = "Please summarize the following paper in under 5 sentences:\n\n"
    
    def summarize_papers(self, papers):
        summary_list = []        
        for i, paper in enumerate(papers):
            # restricting the no. of prompts to 3 to avoid the api limit for Free Users.
            if i < 3:
                each_prompt = self.prompt + paper
                summary_list.append(self.get_response_to_prompt(each_prompt))
        return summary_list

    def get_response_to_prompt(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model= self.model,
                messages=[
                    {"role": self.role, "content": prompt}
                ])
        except Exception as e:
            print("Error in getting response from GPT3: ", e)
            return
        return response.choices[0].message["content"]

