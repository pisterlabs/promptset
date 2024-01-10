import openai
from openai.error import RateLimitError, APIError, APIConnectionError
from tools.secret_squirrel import SecretSquirrel
from time import sleep


class ChatGptConnector():
	
	def __init__(self, requested_model='gpt-3.5-turbo', temperature=0.5):
		"""
		Takes in parameters and grabs credentials
		requested_model - openAI model to be used (default: gpt-3.5-turbo)
		temperature - the tempurature setting for the model (default: 0.5)
		"""
		self.requested_model = requested_model
		self.temperature = temperature
		self.creds = SecretSquirrel().stash
		self.endpoint = "https://api.openai.com/v1/completions"
		openai.api_key = self.creds['open_ai_api_key']
		self.sleep_time = 0.5
		self.sleep_threshold = 15


	def send_message(self, conversation):
		"""
		Sends message to ChatGPT with provided settings
		Has incrementing fall-off for retries incase the API is being slow and stupid
		Returns a Tuple: (success:bool, message:string, error:string)
		"""
		try:
			response = openai.ChatCompletion.create(
				model=self.requested_model,
				messages = conversation.messages,
				temperature = self.temperature
			)
			return (True, response['choices'][0]['message']['content'], None)
		except (RateLimitError, APIError, APIConnectionError) as e:
			print("...", end="\r")
			self.sleep_time *= 2
			if self.sleep_time > self.sleep_threshold:
				return (False, 'the servers are unable to handle requests, currently, and seem to be unreachable', repr(e))
			sleep(self.sleep_time)
			return self.send_message(conversation)
		except Exception as e:
			return (False, 'something seems to have happened, but I am unable to determine what', repr(e))

		