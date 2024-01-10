import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.document_loaders import UnstructuredMarkdownLoader
import tiktoken
from time import sleep

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_model = "gpt-4-1106-preview"
openai_model_max_tokens = 3000

def report_tokens(prompt, model):
    encoded_prompt = tiktoken.encoding_for_model(model).encode(prompt)
    token_count = len(encoded_prompt)
    print(f"\033[37m{token_count} tokens\033[0m in prompt: \033[92m{prompt[:50]}\033[0m")

class SkillMatcher:
    def __init__(self, client, model=openai_model, max_tokens=openai_model_max_tokens):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def load_markdown_content(self, file_path):
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
        texts = [doc.page_content for doc in documents]
        return ' '.join(texts)

    def generate_prompt(self, job_requirements, profile_metadata, role):
        role_messages = {
            'client': 'why this candidate is a good match for the job',
            'talent': 'explain to the talent, why this job opportunity is good for their career development and how to put their skills in practive'
        }
        return (
            f"Job Requirements:\n{job_requirements}\n\n"
            f"Candidate Profile:\n{profile_metadata}\n\n"
            f"Explain in a convincing way {role_messages[role]}:"
        )

    def generate_response(self, system_prompt, user_prompt):
        report_tokens(system_prompt, self.model)
        report_tokens(user_prompt, self.model)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }

        try:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Failed to generate response. Error: ", e)
            return "No response generated"

def generate_explanation(skill_matcher, job_requirements, profile, role):
    system_prompt = "Please generate a detailed explanation for the following:"
    user_prompt = skill_matcher.generate_prompt(job_requirements, profile, role)
    return skill_matcher.generate_response(system_prompt, user_prompt)

def main():
    skill_matcher = SkillMatcher(client)

    job_requirements = skill_matcher.load_markdown_content('./requirements.md')
    recommended_profiles = skill_matcher.load_markdown_content('./recommender.md')

    profiles = recommended_profiles.split('-' * 50 + "\n\n")

    for profile in profiles:
        prompt_job_requirements = job_requirements[:500]
        prompt_profile = profile[:3000]

        explanation_for_client = generate_explanation(skill_matcher, prompt_job_requirements, prompt_profile, 'client')
        print(f"Explanation for client:\n{explanation_for_client}\n")

        explanation_for_talent = generate_explanation(skill_matcher, prompt_job_requirements, prompt_profile, 'talent')
        print(f"Explanation for talent:\n{explanation_for_talent}\n")

if __name__ == "__main__":
    main()
