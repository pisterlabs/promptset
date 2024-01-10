import openai

# Class used to call the OpenAI APi and access GPT-3.
class API():
	"""
	Constructor.

	@priv_key_path Path of the private key needed to access the API.
	@engine OpenAI engine to use

	"""
	def __init__(self, priv_key_path='openai_private_key.priv', engine="text-davinci-003"):
		self._authenticate(priv_key_path)

		self.engine = engine

	def _authenticate(self, priv_key_path):
		with open(priv_key_path) as f:
			openai.api_key = f.read()

	"""
	Complete the text given by @prompt.

	@temperature Value between 0 and 1 controlling the stochasticity (the higher it is, the more "creative" the model is)
	@max_tokens Maximum number of tokens in the answer
	@stop When this token is generated, the model stops generating text. Note that this token is deleted from the answer
	@prompt_end What to add to the end of @prompt, before passing it to GPT-3.
	@response_lstrip Apply lstrip() with this string to the response given by GPT-3.
	@response_rstrip Apply rstrip() with this string to the response given by GPT-3.
	@response_end Add this string to the beginning of the response given by GPT-3, after applying lstrip() and rstrip()
	@response_end Add this string to the end of the response given by GPT-3, after applying lstrip() and rstrip()
	"""
	def complete_text(self, prompt, temperature=0.8, max_tokens=100, stop=None, 
					  prompt_end="\n\n", response_lstrip="", response_rstrip="", response_beginning="", response_end=""):
		# Remove spaces at the beginning and ending (e.g.: "  hi " -> "hi")
		prompt = prompt.strip()

		# Add break line at the end
		prompt += prompt_end

		response = openai.Completion.create(
				prompt=prompt,
		        engine=self.engine,
		        max_tokens=max_tokens,
		        temperature=temperature,
		        stop=stop)

		response_text = response.choices[0].text.lstrip(response_lstrip).rstrip(response_rstrip)
		response_text = response_beginning + response_text + response_end

		return response_text


# > Execute the app in the terminal
if __name__ == "__main__":
	# Complete the user's prompts until it says "EXIT"

	print("----------- text_generation_demo_v1 -----------")
	print("Esta demo utiliza GPT-3 para completar el texto introducido por el usuario.")
	print("En caso de querer salir del programa, escribe 'EXIT'")

	api = API()

	end = False

	while not end:
		user_prompt = input("<User>: ")

		if user_prompt.upper() == "EXIT":
			end = True
		else:
			openai_response = api.complete_text(user_prompt)

			print("<GPT-3>:", openai_response)
