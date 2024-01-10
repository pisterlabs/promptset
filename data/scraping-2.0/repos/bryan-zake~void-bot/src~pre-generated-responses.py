import openai
import os
import time

openai.organization = os.environ.get("OPENAI_ORGANIZATION_ID")
openai.api_key = os.environ.get("OPENAI_API_KEY")
discord_token = os.environ.get("DISCORD_TOKEN")

def bot_response():
	resp = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[
			{"role": "system", "content": "You are the lord of the void."},
			{"role": "user", "content": "Write a one line poem about deleting a message"},
		],
        temperature=0.7
	)
	return resp['choices'][0]['message']['content']

with open("pregenerated_responses.txt", "a") as f:
    for i in range(100):
        try:
            resp = bot_response()
        except openai.error.RateLimitError:
            print("Ratelimiterror")
            time.sleep(1)
        print(resp)
        f.write(f"{resp}\n")
