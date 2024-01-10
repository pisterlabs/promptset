import os
from openai import OpenAI
from serpapi import GoogleSearch
from dotenv import load_dotenv
import json

import logging
from app.azure_gpt import get_client

logging.basicConfig(level=logging.INFO)

load_dotenv()


class OpenAIClient:
    def __init__(self):
        self.client = get_client()

    def _create_chat_completion(
        self, messages, model="gpt-3.5-turbo-1106", return_json=False, seed=420
    ):
        try:
            format_option = {"type": "json_object"} if return_json else None
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                seed=seed,
                response_format=format_option,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error creating chat completion: {e}")
            raise

    def _generic_chat_completion(
        self, system_message, user_message, return_json=False, seed=None, gpt=3
    ):
        if gpt == 3:
            model = "gpt-3.5-turbo-1106"
        elif gpt == 4:
            model = "gpt-4-1106-preview"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        print("generating chat completion")
        return self._create_chat_completion(
            messages=messages,
            return_json=return_json,
            seed=seed,
            model=model,
        )

    def summarize_job(self, description):
        system_message = "You are a helpful assistant. Please summarize the following job description in JSON format. Try to break down the real-world requirements of the job in a structured way."
        user_message = f"Here is the job description: {description}"
        return self._generic_chat_completion(
            system_message, user_message, return_json=True
        )

    def summarize_cv(self, cv):
        system_message = "You are a helpful assistant. Please summarize the following CV in JSON format. Try to break down the real-world requirements of the job in a structured way."
        user_message = f"Here is the CV: {cv}"
        return self._generic_chat_completion(
            system_message, user_message, return_json=True
        )

    def cv_job_match(self, cv, job_description):
        system_message = (
            "You are a helpful assistant. "
            "Your task is to evaluate the match between a CV and a job description. "
            "Consider skills, experience, qualifications, and cultural fit. "
            "Provide a score for each category and an overall score."
            "Be harsh, the job market is competitive, if the candidate has missing skills they are not going to get the job!"
            "Return a JSON object with the scores for each category and an overall score along with some feedback for the candidate."
        )
        user_message = f"Please evaluate the following CV against the job description: {job_description}\n\nCV: {cv}"
        return self._generic_chat_completion(
            system_message, user_message, return_json=True, gpt=4
        )

    def extract_skills_from_cv(self, cv):
        example_response = {"skills": ["Python", "Flask", "Docker"]}
        system_message = f"You are a helpful assistant. Please extract and list the top 5 skills from the following CV. You MUST return a JSON object in the following format: {json.dumps(example_response)}"
        user_message = f"Here is the CV: {cv}"

        response = self._generic_chat_completion(
            system_message, user_message, return_json=True
        )

        skills = json.loads(response)
        if not (
            isinstance(skills, dict)
            and "skills" in skills
            and isinstance(skills["skills"], list)
        ):
            raise ValueError(
                "The response format is incorrect. Expected a dictionary with a key 'skills' containing a list."
            )
        return ", ".join(skills["skills"])


class SerpAPIClient:
    def __init__(self):
        self.serpapi_key = os.getenv("SERP_API_KEY")
        if not self.serpapi_key:
            raise EnvironmentError("Please set SERP_API_KEY environment variable.")

    def search_jobs(self, query, location, chips_filters=None):
        try:
            chips = "date_posted:week"

            # add chips filters
            if chips_filters:
                chips += f", {chips_filters}"

            search = GoogleSearch(
                {
                    "engine": "google_jobs",
                    "q": query,
                    "location": location,
                    "hl": "en",
                    "api_key": self.serpapi_key,
                    "chips": chips,
                    "sort_by": "date",
                }
            )
            return search.get_dict().get("jobs_results", [])
        except Exception as e:
            raise e

    def get_apply_link(self, job_id: str) -> str:
        search = GoogleSearch(
            {
                "engine": "google_jobs_listing",
                "q": job_id,
                "api_key": os.environ.get("SERP_API_KEY"),
            }
        )
        return search.get_dict().get("apply_options", [{}])[0].get("link", "")
