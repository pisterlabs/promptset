import openai 
import ast
class APIHandler():
	def __init__(self,desc,api_key):
		self.stateless_prompt=open("Stateless.txt","r").read().replace("{$}",desc)
		self.desc=desc
		self.api_key=api_key
	def get_response(self):
		openai.api_key=self.api_key
		self.response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=self.stateless_prompt,
            max_tokens=3000,
			temperature=0.3,
			top_p=1,
			frequency_penalty=0,
			presence_penalty=0
        )
	def clean_response(self):
		self.text=self.response.choices[0].text
	def generate_updates(self):
		self.updates=ast.literal_eval(self.text)