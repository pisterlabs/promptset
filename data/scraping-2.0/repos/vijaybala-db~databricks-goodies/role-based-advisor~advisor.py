import os, json
from langchain.llms import Databricks, OpenAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

class RoleBasedAdvisor:
    def __init__(self, language_model='openai', config_file_path=None):
        self.template_string = """{role_name} \
Respond to the user question that is delimited in triple backticks \
with thoughtful and concise instructions that the user can easily implement in their \
day to day life.
user_question: ```{user_question}```
"""
        self.role_description = {}
        self.role_description['doctor'] = """You are a doctor (primary care physician) with 25 years of experience practicing in California. \
You emphasize the importance of a healthy lifestyle that includes nutritious food and vigorous exercise."""
        self.role_description['father'] = """You are the user's father and cares deeply about their well being. You emphasize the importance of \
working hard and getting a good education."""
        self.role_description['business_partner'] = """You are the user's business partner. You share a mutual interest in the success of your \
company. You emphasize actions that will maximize the long term viability and profitability of the company and achieving its mission."""
        self.role_description['career_coach'] = """You are the user's manager at work. You see great potential in the user to progress in their \
career. You emphasize actions that maximize the user's chances for a promotion and continue their trajectory to become a senior executive."""
        self.user_question = "I want to live a life that maximizes happiness and creates a positive impact on the world. What \
are the top 5 things I should do in the next week towards these goals?"

        self.language_model = language_model
        if config_file_path is not None:
            with open(config_file_path) as f:
                self.config = json.load(f)
        self.llm = self.get_llm(language_model)

    def get_llm(self, language_model='openai'):
        load_dotenv()
        if 'DATABRICKS_RUNTIME_VERSION' in os.environ and language_model == 'openai':  # Running in Databricks
            if 'OPENAI_API_KEY' not in os.environ:
                os.environ['OPENAI_API_KEY'] = dbutils.secrets.get('vbalasu', 'openai-databricks')

        if language_model == 'openai':
            llm = OpenAI(temperature=0.0, max_tokens=500)
            return llm
        elif language_model == 'llamav2':
            llm = Databricks(cluster_driver_port=self.config['port'], cluster_id=self.config['cluster_id'],
                        model_kwargs={'temperature':0.0, 'max_new_tokens':500})
            return llm
        else:
            print('Unknown language model')
            return False
        
    def answer_as_role(self, user_question, role, verbose=False):
        prompt_template = ChatPromptTemplate.from_template(self.template_string)
        prompt = prompt_template.format_prompt(role_name=role, user_question=user_question)
        question = prompt.messages[0].content
        if verbose:
            print('/*\n', f'LANGUAGE MODEL: {self.language_model}\n\n', question, '*/\n\n')
        return self.llm(question)