from data.chatbot.tests import mock
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
load_dotenv('../../.env')

# True to use OpenAI api, False to send mock data
GPT = False

SYSTEM_MESSAGE_PROMPT = """
        You are a job recruiter who will be interviewing an applicant.
        You must respond in first person point of view. You are directly talking to the applicant, no one else.
        You have access to information about the interview. Provided below is the outline of the interview information:
        c_name: the name of your company.
        business: the business type of your company.
        description: the description of your company.
        desired position of applicant: the desired position of the applicant.
        programming skills of applicant: the programming languages the applicant is most confident in.
        In addition, you have access to important details from the resume of the applicant.
        """

# TODO implement grading/pass algorithm
GENERATE_INTERVIEW_PROMPT = """
        Your task is to generate an interview question. I will provide both interview information and resume details below.
        """
EVALUATE_RESPONSE_PROMPT = """
        Your task is to evaluate the response of the user, based on the question you have asked. You must respond exactly in this format: 
        First, start with Strengths, and list out the strengths of the response.
        Second, start with Areas for Improvement, and list out the areas for improvement of the response.
        Each list item must begin with a hyphen, and end with a newline.
        """


class InterviewBot:

    def __init__(self) -> None:
        self.interviewer = OpenAI(api_key=os.getenv('OPENAI_KEY'))

    def generate_question(self, resume_contexts, resume_embeddings, interview_info, interview_info_embeddings, message_history):
        """
        Generate interview question.
        """
        if not GPT:
            return (mock()["starter_message"])

        # embeddings as np array
        doc_emb = np.asarray(resume_embeddings)
        query_emb = np.asarray(interview_info_embeddings)
        query_emb.shape

        # compute dot product between both embeddings
        scores = np.dot(query_emb, doc_emb.T)[0]

        # Find the highest scores
        max_idx = np.argsort(-scores)
        most_relevant_contexts = []
        top_k = 5
        for idx in max_idx[0:top_k]:
            most_relevant_contexts.append(resume_contexts[idx])

        # test
        # print(most_relevant_contexts)

        passages = "\n".join(most_relevant_contexts)

        # prompt the GPT
        evaluation_system_config = [
            {
                "role": "system",
                "content": f"{SYSTEM_MESSAGE_PROMPT} {GENERATE_INTERVIEW_PROMPT}"
            }
        ]

        initial_prompt = f"Here is the interview information: {interview_info}. Here are the important details from the resume of the applicant: {passages}. Generate an interview question for the applicant."

        updated_history = evaluation_system_config + message_history + [{
            "role": "user", "content": initial_prompt
        }]
        print(updated_history)

        response = self.interviewer.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=updated_history
        )
        return (response.choices[0].message.content)

    def evaluate_response(self, message_history, user_input):
        """
        Evaluate the response of the user. Message history contains
        """
        if not GPT:
            return (mock()["evaluation_message"])

        evaluation_system_config = [
            {
                "role": "system",
                "content": f"{SYSTEM_MESSAGE_PROMPT} {EVALUATE_RESPONSE_PROMPT}"
            }
        ]
        updated_history = evaluation_system_config + message_history + [{
            "role": "user", "content": user_input
        }]

        response = self.interviewer.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=updated_history
        )

        # parse the response of the user
        return (response.choices[0].message.content)
