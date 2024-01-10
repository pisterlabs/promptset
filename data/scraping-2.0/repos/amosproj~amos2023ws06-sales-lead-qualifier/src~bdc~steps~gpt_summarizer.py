# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 Berkay Bozkurt <resitberkaybozkurt@gmail.com>
# SPDX-FileCopyrightText: 2023 Sophie Heasman <sophieheasmann@gmail.com>


import time
from http import HTTPStatus

import openai
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas import DataFrame
from requests import RequestException
from tqdm import tqdm

from bdc.steps.step import Step, StepError
from config import OPEN_AI_API_KEY
from database import get_database
from logger import get_logger

log = get_logger()


class GPTSummarizer(Step):
    """
    The GPTSummarizer step will attempt to download a businesses website in raw html format and pass this information
    to OpenAIs GPT, which will then attempt to summarize the raw contents and extract valuable information for a
    salesperson.

    Attributes:
        name: Name of this step, used for logging
        added_cols: List of fields that will be added to the main dataframe by executing this step
        required_cols: List of fields that are required to be existent in the input dataframe before performing this
            step
    """

    name = "GPT-Summarizer"
    model = "gpt-4"
    no_answer = "None"

    # system and user messages to be used for creating company summary for lead using website.
    system_message_for_website_summary = f"You are html summarizer, you being provided the companies' htmls and you answer with the summary of three to five sentences including all the necessary information which might be useful for salesperson. If no html then just answer with '{no_answer}'"
    user_message_for_website_summary = (
        "Give salesperson a summary using following html: {}"
    )

    extracted_col_name_website_summary = "sales_person_summary"
    gpt_required_fields = {
        "website": "google_places_detailed_website",
        "place_id": "google_places_place_id",
    }

    added_cols = [extracted_col_name_website_summary]
    required_cols = gpt_required_fields.values()

    client = None

    def load_data(self) -> None:
        self.client = openai.OpenAI(api_key=OPEN_AI_API_KEY)

    def verify(self) -> bool:
        if OPEN_AI_API_KEY is None:
            raise StepError("An API key for openAI is need to run this step!")
        return super().verify()

    def run(self) -> DataFrame:
        tqdm.pandas(desc="Summarizing the website of leads")
        self.df[self.extracted_col_name_website_summary] = self.df.progress_apply(
            lambda lead: self.summarize_the_company_website(
                lead[self.gpt_required_fields["website"]],
                lead[self.gpt_required_fields["place_id"]],
            ),
            axis=1,
        )
        return self.df

    def finish(self) -> None:
        pass

    def summarize_the_company_website(self, website, place_id):
        """
        Summarise client website using GPT. Handles exceptions that mightarise from the API call.
        """

        if website is None or pd.isna(website):
            return None
        company_summary = get_database().fetch_gpt_result(place_id, self.name)
        if company_summary:
            return company_summary["result"]

        html = self.extract_the_raw_html_and_parse(website)

        if html is None:
            return None
        max_retries = 5  # Maximum number of retries
        retry_delay = 5  # Initial delay in seconds (5 seconds)

        for attempt in range(max_retries):
            try:
                log.info(f"Attempt {attempt+1} of {max_retries}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_message_for_website_summary,
                        },
                        {
                            "role": "user",
                            "content": self.user_message_for_website_summary.format(
                                html
                            ),
                        },
                    ],
                    temperature=0,
                )

                # Check if the response contains the expected data
                if response.choices[0].message.content:
                    company_summary = response.choices[0].message.content

                    if company_summary == self.no_answer:
                        return None
                    get_database().save_gpt_result(company_summary, place_id, self.name)
                    return company_summary
                else:
                    log.info("No summary data found in the response.")
                    return None
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    log.warning(
                        f"Rate limit exceeded, retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    log.error("Max retries reached. Unable to complete the request.")
                    break
            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.BadRequestError,
                openai.AuthenticationError,
                openai.PermissionDeniedError,
                Exception,
            ) as e:
                # Handle possible errors
                log.error(
                    f"An error occurred during summarizing the lead with GPT: {e}"
                )
                pass

    def extract_the_raw_html_and_parse(self, url):
        try:
            # Send a request to the URL
            response = requests.get(url)
        except RequestException as e:
            log.error(f"An error occured during getting repsonse from url: {e}")
            return None

        # If the request was successful
        if not response.status_code == HTTPStatus.OK:
            log.error(f"Failed to fetch data. Status code: {response.status_code}")
            return None
        try:
            # Use the detected encoding to decode the response content
            soup = BeautifulSoup(response.content, "html.parser")

            texts = []
            for element in soup.find_all(["h1", "h2", "h3", "p", "li"]):
                texts.append(element.get_text(strip=True))
            return " ".join(texts)
        except UnicodeDecodeError as e:
            return None
