from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

import os
from dotenv import load_dotenv

load_dotenv()

google_cse_id = os.getenv("GOOGLE_CSE_ID")
google_api_key = os.getenv("GOOGLE_API_KEY")

GOOGLE_CSE_ID=google_cse_id
GOOGLE_API_KEY=google_api_key

search = GoogleSearchAPIWrapper()

tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
)

tool.run("Obama's first name?")

