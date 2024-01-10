import os
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
from os.path import join, dirname
from dotenv import load_dotenv
import asyncio

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class InterviewLLM:
     def __init__(self, job_title, job_description, cv) -> None:
          self.job_title = job_title
          self.job_description = job_description
          self.cv = cv
          self.llm = OpenAI(temperature=0.6, max_tokens=-1, openai_api_key=OPENAI_API_KEY)

     def generate_questions(self):
          prompt1 = """
          You are a recruiter for a company. You are recruiting for the role of {job_title}. 
          ---
          The description of the job is as follows: 
          {job_description}
          ---
          The applicant's CV is as follows: {CV}
          ---
          You have to generate SCREENING QUESTIONS for the applicant based on the CV and the job description.
          These questions must effectively assess their suitability for the role of {job_title}
          Make sure to RETURN ONLY JSON, not code or any other text. 
          """

          prompt1_template = PromptTemplate(
               input_variables=["job_title" ,"job_description", "CV"],
               template= prompt1
          )

          chain1 = LLMChain(llm = self.llm, prompt=prompt1_template, 
                            output_key="dictionary"
                        )
          
          return chain1.run(job_title=self.job_title, job_description=self.job_description, CV=self.cv)
     
     @staticmethod
     async def async_gen(sqc, inputs):
          resp = await sqc.arun(inputs)
          return resp
     
     @staticmethod
     async def evaluate_answers(job_title, questions, answers):
          llm = OpenAI(temperature=0.6, max_tokens=-1, openai_api_key=OPENAI_API_KEY)

          eval_prompt = PromptTemplate(
               input_variables=["job_title", "question", "answer"],
               template="""
               You are a recruiter for a company. You are recruiting for the role of {job_title}. 
               You have provided an interview question to candidate and they have given you the answer.
               Your job is to see how correct the provided answer is and evaluate the answer and score it out of 10. You are allowed to use decimal values.
               ---
               The question given to candidate:
               {question}
               ---
               The answer provided by the candidate:
               {answer}
               ---
               Evaluate the answer Only return the score out of 10 and no other text.
               """
          )

          eval_chain = LLMChain(llm=llm,
                                   prompt=eval_prompt,
                                   output_key="score")

          sqc = SequentialChain(chains=[eval_chain],
                                   input_variables=["job_title", "question", "answer"],
                                   output_variables=["score"],
                                   verbose=False)

          tasks = []
          for q, a in zip(questions, answers):
               tasks.append(InterviewLLM.async_gen(sqc, {"job_title": job_title,
                                                            "question": q, "answer": a}))
          results = await asyncio.gather(*tasks)
          score_per_question = [float(i) for i in results]
          sum_score = sum(score_per_question)

          return score_per_question, sum_score

          







          
          
     


     


