import os
import dotenv
import openai

dotenv.load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

response = openai.Completion.create(
    engine="text-davinci-002",
    prompt="What is the capital of France?",
    max_tokens=1024,
    # temperature=0,
    # top_p=1,
    # n=1,
    # stream=False,
    # logprobs=None,
    # stop="\n",
)
print(response)
