import subprocess
import sys
from abc import ABC, abstractmethod
import openai

def copy_to_pasteboard(text: str) -> None:
	"""
	Copy the provided text to the macOS pasteboard using pbcopy.
	"""
	process = subprocess.Popen('pbcopy', universal_newlines=True, stdin=subprocess.PIPE)
	process.communicate(text)
	process.wait()
	
class Querier:
	@abstractmethod
	def performQuery(self, prompt):
		pass

class HumanQuerier(Querier):
	def performQuery(self, prompt):
		print(prompt)
		copy_to_pasteboard(prompt)
		lines = []
		try:
			for line in sys.stdin:
				lines.append(line)
		except EOFError:
			pass
		print(lines)
		return "".join(lines)

class OpenAIQuerier(Querier):
	def __init__(self, model):
		self.__model = model
		pass
		
	def performQuery(self, prompt):
		prompt_content = f"{prompt}"
		print(prompt_content)
		# Send the prompt to the OpenAI API
		# This assumes that you have the OPENAI_API_KEY environment variable set
		response = openai.Completion.create(
			engine=self.__model,
			prompt=prompt_content,
			max_tokens=1000
		)
		
		# Extract and print the generated code
		return response.choices[0].text.strip()
		
