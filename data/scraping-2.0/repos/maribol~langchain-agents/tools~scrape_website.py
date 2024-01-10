
# Import necessary modules
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from bs4 import BeautifulSoup
from typing import Type
import requests
import json
import os

# Load environmental variables
load_dotenv()

# Get browserless API key from environmental variables
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")

def scrape_website(objective: str, url: str):
    """Scrape appropriate data from website url."""

    # Establish headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Establish data for the request
    data = {"url": url}
    data_json = json.dumps(data)

    # Set post url and send post request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Parse response, return text or error message
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        return (f"Could not scrape this website. Got status code {response.status_code}.")


class ScrapeWebsiteInput(BaseModel):
    """Define inputs schema for scrape_website function"""

    objective: str = Field(description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    """Define a class for the 'scrape_website' tool"""

    name = "scrape_website"
    description = "Useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        """Run the scrape website function when called."""

        return scrape_website(objective, url)

    def _arun(self, url: str):
        """Raise an error if _arun method is called as it is not implemented."""

        raise NotImplementedError("This method has not been implemented.")