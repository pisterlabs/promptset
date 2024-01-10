from openai import OpenAI

class AskGPT:
	"""
	A class that creates a response using a specified OpenAI GPT model.
	A response is created depending on the user's speech, the bot's role, the prompt used,
 	and the conversation history.
	"""
	
	def __init__(self, api_keys:dict, setting_objects:dict):
     
		self.client = OpenAI(api_key = api_keys['OPENAI-API-KEY'])
  
		self.conversation_history = []
  
		# get gpt model
		profile_settings = setting_objects['profile_settings']
		self.model = profile_settings.retrieve_property('gpt_model')
		if self.model in api_keys:
			self.model = api_keys[self.model]
   
		self._Load_in_settings(setting_objects)
		self.system_message = self._construct_system_message()
		
	def ask_GPT(self, speech:str, manual_request:bool=False, max_tokens:int=100) -> str:
		"""
		Uses the user's speech, the bot's role, and the conversation history 
		to create a response using OpenAI's GPT model.
		"""
		# get response from gpt
		response = self._send_gpt_request(speech, manual_request, max_tokens)
		# cleanup response
		response = self._clean_response(response, self.entity_name)
	
		return response

	def _send_gpt_request(self, speech:str, manual_request:bool, max_tokens:int) -> str:
		"""
		Sends a POST request to the GPT model and returns the response.
		"""
		messages = []

		# For manual requests, we start with the system message and add the user message
		if manual_request:
			messages.append({"role": "assistant", "content": self.system_message})
			messages.append({"role": "user", "content": speech})
		else:
			self._update_conversation("user", speech)
			messages = self.conversation_history

		response = self.client.chat.completions.create(
			model=self.model,
			messages=messages,
			max_tokens=max_tokens
		)

		# Extract the message
		response = response.choices[0].message.content

		return response 

	def _update_conversation(self, role:str, content:str) -> None:
		"""
  		Updates the conversation history with the user's speech and the bot's response.
    	"""
		if role == "user":
			self.conversation_history.append({"role": "assistant", "content": self.system_message})
		self.conversation_history.append({"role": "user", "content": content})
  
	def _update_prompt(self, prompt:str) -> None:
		"""Updates the prompt to be used for the GPT model."""
		self.prompt = prompt
  
	def _clean_response(self, response:str, bot_name:str) -> str:
		"""
  		Sometimes the response from the GPT model will incorrectly include the bot's name in the response.
    	"""
  
		bad_inputs = [f'{bot_name} said: ', f' {bot_name} said: ']
		for example in bad_inputs:
			if example.startswith(response):
				response = response.replace(example, '')
		return response.strip()

	def _construct_system_message(self) -> str:
		"""
		Constructs a system message that is used to initialize the conversation history.
		"""
		system_message = f"You are a {self.personality} virtual assistant named {self.entity_name} who speaks {self.language}. Your role is {self.role}."
  
		if self.persona:
			system_message += f" Take on the persona of {self.persona}. Engage with the user as if you are {self.persona}."
   
		if self.user_name:
			system_message += f" The user's name is {self.user_name}."
   
		return system_message
     
     
	def _Load_in_settings(self, setting_objects:dict) -> None: 
		"""
		Loads in the profile settings from the appropriate file.
		"""
		self.master_settings = setting_objects['master_settings']
		self.profile_settings = setting_objects['profile_settings']
		self.profile_name = self.master_settings.retrieve_property('profile')
		self.entity_name = self.profile_settings.retrieve_property('name', self.profile_name)
		self.language = self.profile_settings.retrieve_property('language', self.profile_name)
		self.personality = self.profile_settings.retrieve_property('personality', self.profile_name)
		self.persona = self.profile_settings.retrieve_property('persona', self.profile_name)
		self.role = self.profile_settings.retrieve_property('role', self.profile_name)
		self.gpt_model = self.profile_settings.retrieve_property('gpt_model', self.profile_name)
		#self.user_name = self.profile_settings.retrieve_property('user_name', self.profile_name)	
		self.user_name = None