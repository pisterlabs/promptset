from data.chatbot.tests import mock
import os
from openai import OpenAI
from dotenv import load_dotenv
from data.chatbot.resume_helper import ResumeHelper

load_dotenv('../../.env')

# True to use OpenAI api, False to send mock data
GPT = False

SYSTEM_MESSAGE_PROMPT = """
        TODO
        """

RESUME_ANALYSIS_PROMPT = """
        TODO
        """


class ResumeBot:

    def __init__(self) -> None:
        self.interviewer = OpenAI(api_key=os.getenv('OPENAI_KEY'))
        self.resume_helper = ResumeHelper()

    def generate_analysis(self, resume_contexts, resume_embeddings, interview_info, interview_info_embeddings, message_history):
        """
        Generate analysis of resume.
        """
        if not GPT:
            return (mock()["ordered_content"])

        groups = self.resume_helper.group(resume_contexts, interview_info)
        
        response = "\n///BREAK///\n".join(groups)

        return response
