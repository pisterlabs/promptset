import openai
import json
import sys

with open("D:\\twitter-clone-new\\config.json") as f:
    config = json.load(f)
    OPENAI_API_KEY = config["OPENAI_API_KEY"]

openai.api_key = OPENAI_API_KEY
user_response = " ".join(sys.argv[1:])

twitter_writer_prompt = (
    "You are going to be Twitter writer. "
    "Here is my idea, about which I would like to write. "
    "Your main goal is to write me a tweet which is going to be viral. "
    "Style of text should be polite. Max tweet characters is 100. "
    "Do not write any comments to tweet, only tweet text. Idea: "
)


def create_tweet(text: str) -> str:
    prompt = twitter_writer_prompt + text

    openai_response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0
    )

    result = openai_response.choices[0].text
    return result.strip()


result = create_tweet(user_response)
print(result)
