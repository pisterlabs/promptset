import os
import logging
import argparse

from datetime import datetime

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.together import Together
from langchain_core.callbacks import StreamingStdOutCallbackHandler, CallbackManager
from prompts import resume_checker_prompt, check_output_parser, resume_cover_letter_prompt

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("resume_logger.log"),
        logging.StreamHandler()
    ]
)


def trim_space(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ")


class ResumeAnalyser:
    """ ResumeAnalyser takes in a path to your resume and a job posting URL and lets you know if you're fit for the
    job or not"""
    llm = None
    resume_content: str = ""
    job_content: str = ""

    def __init__(self, resume_path: str, job_posting_url: str):
        self.resume_path = resume_path
        self.job_posting_url = job_posting_url

        model_name = os.getenv("LLM_MODEL")
        if model_name == "openai":
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
        elif model_name == "ollama":
            self.llm = Ollama(
                model="mistral",
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )
        elif model_name == "together":
            self.llm = Together(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.7,
                top_k=1,
            )
        else:
            raise ValueError("Invalid LLM model name")

        self.load_resume(self.resume_path)
        self.load_job_site(self.job_posting_url)

    def load_resume(self, path):
        """ loads the resume text from the resume path """
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            self.resume_content += trim_space(doc.page_content)

    def load_job_site(self, url):
        """ loads the job text from the job site url """
        loader = WebBaseLoader(url)
        docs = loader.load()
        for doc in docs:
            self.job_content += trim_space(doc.page_content)

    def run(self):
        """ Runs the program """
        check_prompt_str = resume_checker_prompt.format(resume=self.resume_content, job_description=self.job_content)
        logging.debug(check_prompt_str)
        output = self.llm.predict(check_prompt_str)
        result = check_output_parser.parse(output)
        score, value, explanation, fixes = result["score"], result["value"], result["explanation"], result["fixes"]
        print(f"""
Job fit: {value}
Explanation: {explanation}
Your Resume Score is: {score}
            """)
        if value is True:
            self.generate_cover_letter()
        else:
            print(f"""
You're not fit for this role, but here's a possible way to improve your resume:
{fixes}
                """)

    def generate_cover_letter(self):
        """ generate cover letter for the job"""
        cover_letter_prompt_str = resume_cover_letter_prompt.format(resume=self.resume_content,
                                                                    job_description=self.job_content)
        cover_letter = self.llm.predict(cover_letter_prompt_str)
        logging.debug(cover_letter_prompt_str)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"cover_letter_{timestamp}.txt"
        filename = os.path.join("cover_letters", filename)

        with open(filename, "w") as f:
            f.write(cover_letter)

        logging.info(f"Cover letter generated at {filename}")
        print(f"""
Here's an example cover letter you can use
{cover_letter}
        """)
