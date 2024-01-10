import openai
import dotenv
import os

dirname = os.path.abspath(os.path.dirname(__file__))
dotenv.load_dotenv(os.path.join(dirname, ".env"))
api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
api_base = os.getenv("API_BASE")
model = os.getenv("MODEL")

API_BASE = "https://api.pawan.krd/v1"  # https://api.openai.com/v1
HEADERS = {
	"Authorization": "Bearer " + api_key,
	"Content-Type": "application/json"
}

openai.api_key = api_key
openai.api_base = api_base


def ask(message, system = "", model=model):
	completion = openai.ChatCompletion.create(
		model=model,
		messages=[
			{
				"role": "system",
				"content": system
			},
			{
					"role": "user",
					"content": message
			}
		]
	)
	return completion.choices[0].message.content
