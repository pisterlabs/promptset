# A function to get a set of question for performing an interview based on a person's Resume.
# The output of Resume_ContentExtractor.py in this repo is the ideal input for this function.
# The result of this function is a set of 10 questions.

from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
# Add your own OPENAI_API_KEY for usage
def generate_questions(resume,role='',experience=''):
    _PROMPT_TEMPLATE = """
    this is the resume of user:
    {resume_details}
    here is the role he want to join in :
    {role}
    Based on the following experience:
    {experience}
    What are your interview  questions for the given user resume and role he want to join in with that experience?
    generate no of  questions = {questions}!
    """
    PROMPT = PromptTemplate(input_variables=["resume_details", "role", "experience",'questions'], template=_PROMPT_TEMPLATE)
  
    llm1 = OpenAI(model_name="text-davinci-003", temperature=0)
    chain = LLMChain(llm=llm1, prompt=PROMPT)
    prompt = chain.predict_and_parse(resume_details= gen_text(resume),
    role= role,
    experience= experience,
      questions=10)
    return prompt.split('\n')
