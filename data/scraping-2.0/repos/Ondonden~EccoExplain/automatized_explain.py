import numpy as np
from prompts import prompt_new
import openai
import pandas as pd
from embedding_searcher import Searcher

class Explainer:
	def __init__(self, nmf_object, api_key):
		openai.api_key = api_key
		self.api_key = api_key
		self.components = nmf_object.components 
		self.tokens = nmf_object.tokens[0]
		self.num_of_tokens = len(nmf_object.tokens[0])
		self.threshold = threshold


# normalize component uses the range/minimum of each factor
	def normalize_components(self, components_lists):
		range_vals = [np.ptp(component) for component in components_lists]
		min_vals = [np.min(component) for component in components_lists]
		# nested list comprehension, which normalizes all values between 0 and 1,
		# multiplies value by 10 and rounds to nearest int.
		normalized_components = [
			np.around([
				int(((x - min_vals[i]) / range_vals[i]) * 10)
				for x in components_lists[i]
			])
			for i in range(len(components_lists))
		]
		return normalized_components

	def create_new_result_string(self, normalized_components, tokens):
		new_strings = []
		factor_count = 1
		for component in normalized_components:
			new_string = f'Factor {factor_count}:\nActivations\n<start>\n'
			for i in range(len(component)):
				new_string = new_string + f'{tokens[i]}\t{component[i]}\n'
			new_string = new_string + '<end>\n'

			new_string = new_string + f'same string but with all zeros filtered out:\n\n<start>\n'	
			for i in range(len(component)):
				if component[i]>0:
					new_string = new_string + f'{tokens[i]}\t{component[i]}\n'
			new_string = new_string +'<end>\n'

			#print(new_string)

			new_strings.append(new_string)
			factor_count += 1
		return new_strings


	def create_prompt(self, string_result):
		prompt_for_gpt = prompt_new.prompt_start.strip()
		searcher = Searcher(string_result)
		prompt_for_gpt = prompt_for_gpt + searcher.add_examples_to_prompt()
		prompt_for_gpt = prompt_for_gpt + string_result
		prompt_for_gpt = prompt_for_gpt + prompt_new.prompt_end.strip()
		return prompt_for_gpt
	

	def get_response(self, prompt):
		# gets response in text using GPT 3.5 based model text-davinci-003
		response = (openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.7, max_tokens=600))["choices"][0]["text"]
		#response = 'test_response'
		return response

	def analyze(self):
		result_string_list = self.create_new_result_string(self.normalize_components(self.components), self.tokens)
		factor_explanations = []
		for i in range(len(result_string_list)):
			prompt = self.create_prompt(result_string_list[i])
			response = self.get_response(prompt)
			factor_explanations.append(response)
		return factor_explanations

	
	#todos:
	#TODO check if model is ok and throw
	#TODO make sure text input isn't too long
	#TODO make api-call to openai try/except like, in case of unreliability/no internet
	#TODO include as many examples as you have space for.
	#TODO Figure out how to use 'gpt-3.5-turbo' instead, since it's 1/10 the price of davinci.
	#TODO write tests for the different methods

	# Instead write: Word: Activation
	# for every word in the sequence
	# OR You can place ** around every word...

