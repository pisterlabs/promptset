from OpenAIAuth import Authenticator
from dotenv import load_dotenv
import os

load_dotenv()

config = os.environ
auth = Authenticator(
        config["OPENAI_EMAIL"],
        config["OPENAI_PASSWORD"],
        )

auth.begin()
token = auth.get_access_token()
print(f"OPENAI_ACCESS_TOKEN={token}")
