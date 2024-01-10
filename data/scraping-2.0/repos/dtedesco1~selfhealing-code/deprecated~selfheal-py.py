import openai
import os
import re

# Set the OpenAI API key
openai.api_key = open("key.txt", "r").read().strip("\n")


suggestions_boilerplate = ""

code_boilerplate = "DO NOT explain what you're doing in natural language. DO NOT use a codeblock. ONLY respond with code. Make sure to avoid using fake libraries, 'example' URLs, dummy API key variables, or dummy data. Your code can only utilize API keys if the user has given them to you. If you encounter any of these, change your approach and, if necessary, your data source. We want to make the code do what the user wants right out of the box!"


# Define the function to generate pseudocode for later prompts
def generate_pseudocode(user_input):
	# Send the user input to GPT-3 to generate the script
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[
			{"role": "system", "content": f"You are an AI programming assistant that gives coding instructions without writing actual code.\n\n- Follow the user's requirements carefully & to the letter.\n- First think step-by-step -- describe your plan for what to build in pseudocode, written out in great detail. Be clear, but as concise as possible. List steps to take, but don't include actual code snippets."},
			{"role": "user", "content": f"How would you write a Python script that does the following:\n{user_input}\nMake sure to avoid using fake libraries, 'example' URLs, or dummy data. We want to make the code do what the user wants right out of the box!"}],
		max_tokens=3000,
		n=1,
		stop=None,
		temperature=0.7
	)
	# Extract the generated script from the API response
	response = response['choices'][0]['message']['content']
	return response

def generate_error_instructions(user_input, prev_response, errors_list):
# Define the function to generate pseudocode for later prompts
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[
			{"role": "system", "content": f"You are an AI programming assistant that gives coding instructions without writing actual code.\n\n- Follow the user's requirements carefully & to the letter.\n- First consider the various things that could be causing the error. Then think step-by-step -- describe your plan for how to address the error. Be clear, but as concise as possible. List steps to take, but don't include actual code snippets."},
			{"role": "user", "content": f"You previously produced a faulty response to the following prompt: 'Write Python code that solves the following challenge:\n{user_input}\nThe most recent faulty response was:\n'''\n{prev_response}\n'''\nThis faulty response produced the following error:\n'''{errors_list[-1]}\nHow would you fix this error while also avoiding previous errors? Do NOT write actual code in your response. Be as brief and concise as possible. Make sure to avoid using fake libraries, 'example' URLs, URLs that do not provide the necessary data, dummy API key variables, or dummy data. Only use API keys if the user has given them to you. If you encounter any of these, change your approach. We want to make the code do what the user wants right out of the box!"}],
		max_tokens=3000,
		n=1,
		stop=None,
		temperature=0.7
	)
	# Extract the generated script from the API response
	response = response['choices'][0]['message']['content']
	return response

# Define the function to generate the python code
def generate_code(user_input, pseudocode):
	# Send the user input to GPT-3 to generate the script
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[
			{"role": "system", "content": "You are a bot that responds with only working, errorless Python code and nothing else."},
			{"role": "user", "content": f"Write Python code that solves the following challenge:\nPrint the word 'foobar'\n\nDO NOT explain what you're doing in natural language. DO NOT use a codeblock. ONLY respond with code. Your code:"},
			{"role": "assistant", "content": "print('foobar')"},
			{"role": "user", "content": "Perfect! Thank you. Let's try another."},
			{"role": "user", "content": f"Write Python code that solves the following challenge:\n{user_input}\n\nHere is some pseudocode to help you:\n{pseudocode}\n\nDO NOT explain what you're doing in natural language. DO NOT use a codeblock. ONLY respond with code. Make sure to avoid using fake libraries, 'example' URLs, URLs that do not provide the necessary data, dummy API key variables, or dummy data. Only use API keys if the user has given them to you. If you encounter any of these, change your approach and, if necessary, your data source. We want to make the code do what the user wants right out of the box! Your code:"}],
		max_tokens=3000,
		n=1,
		stop=None,
		temperature=0.7
	)
	# Extract the generated script from the API response
	response = response['choices'][0]['message']['content']
	return response

# Define the function to run the generated code
def run_code(code):
	exec(code)

# Define the function to fix errors using GPT-3
def fix_error(prev_response, error_instructions, attempt_num, max_attempts):
	if attempt_num >= max_attempts:
		raise Exception("Max number of fix attempts reached. Please fix the error manually.")
	prompt = f"You previously produced a faulty response to the following prompt: 'Write Python code that solves the following user need:\n{user_input}'\n\nThe most recent faulty response was:\n'''\n{prev_response}\n'''\nHere are instructions to address the error: {error_instructions}\nCan you generate a new response so that it meets the goal of its prompt and fixes the error? Be creative to find new solutions. DO NOT explain what you're doing in natural language. DO NOT use a codeblock. ONLY respond with code. Your code:"
	# Send the error message to GPT-3
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[
			{"role": "system", "content": "You are a bot that responds with only working, errorless Python code and nothing else. Your outputs can be pasted directly into a Python file and run. You do not produce code that repeats past errors"},
			{"role": "user", "content": "Write Python code that solves the following challenge:\nPrint the word 'foobar'\nYour code:"},
			{"role": "assistant", "content": "print('foobar')"},
			{"role": "user", "content": "Perfect! Thank you. Let's try another."},
			{"role": "user", "content": "Write Python code that solves the following challenge:\nPrint the numbers 1 through 10'\nYour code:"},
			{"role": "assistant", "content": "print('def print_numbers():\n    for i in range(1, 11):\n        print(i)')"},
			{"role": "user", "content": "Perfect! Thank you. Let's try another."},
			{"role": "user", "content": prompt}],
		max_tokens=3000,
		n=5,
		stop=None,
		temperature=0.7
	)
	response = response['choices'][0]['message']['content']
	if not response:
		raise Exception("Could not find a solution to the error. Please fix it manually.")
	return response


# Define function that returns only True or False for whether or not the code solves the problem
def intent_check(input, prev_response):
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[
			{"role": "system", "content": "You are a bot that determines whether a piece of code solves a user's needs or not. You responds with ONLY 'True' or 'False' and nothing else."},
			{"role": "user", "content": "User input:  '''Write Python code that solves the following challenge:\nPrint the word foobar'''\nCode: '''print('foobar')'''"},
			{"role": "assistant", "content": "True"},
			{"role": "user", "content": "User input:  '''Write Python code that solves the following challenge:\nPrint the numbers 1 through 10'''\nCode:\n'''print('def print_numbers():\n    for i in range(1, 9):\n        print(i)')'''"},
			{"role": "assistant", "content": "False"},
			{"role": "user", "content": "Perfect! Thank you. Let's try another."},
			{"role": "user", "content": f"User input: '''Write Python code that solves the following challenge:\n{input}''' Code: '''{prev_response}'''"}],
		max_tokens=3000,
		n=5,
		stop=None,
		temperature=0.7
	)
	response = response['choices'][0]['message']['content']
	# Convert 'response' to a boolean
	if response == "True":
		response = True
	elif response == "False":
		response = False
	else:
		raise Exception("Could not determine whether or not the code solves the problem.")
	return response

# Define the function to check if the code actually solves the problem
def solve_intent(input, prev_response):
	# Ask the AI if `prev_response` actually solves the problem or not
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[
			{"role": "system", "content": "You are an evaluator that determines whether or not a response to a prompt solves user needs or not. You give clear feedback as to why a response may not meet user needs."},
			{"role": "user", "content": f"Another AI has generated a response to a prompt, but it does not solve the user's needs. Here is the prompt:\nWrite Python code that solves the following challenge:\n{input}\n\nHere is the response:\n'''\n{prev_response}\n'''\nRemind the other AI to avoid using 'example' URLs, dummy API key variables, or data sources that do not actually provide what's needed. Only use API keys if the user has given them to you. We want to make the code do what the user wants right out of the box! How can this code be adapted to solve the user's needs? Use mainly natural language to give suggestions to the AI. Be specific in your suggestions, but be brief."}],
		max_tokens=3000,
		n=1,
		stop=None,
		temperature=0.7
	)
	response = response['choices'][0]['message']['content']
	return response

# Define the main function to run the first script
def main():
	global user_input
	# user_input = input("Please enter what you would like the Python script to do: ")
	# user_input = "Print the numbers 1 to 10"
	# user_input = "Print a fibonacci sequence up to 10, but add a unique line of poetry after each number. Make each line of poetry refer to the number in the sequence with which it's associated.""
	# user_input = "In the terminal, print ASCII characters in the image of a cat's head."
	# user_input = "Print a pattern of asterisks that looks like a Christmas tree."
	# user_input = "Generate a mathematical function whose results look like a flower when printed in the terminal. Print those results in the terminal."
	# user_input = "Generate a mathematical function whose results look like a spiral galaxy when printed in the terminal. Print those results in the terminal."
	user_input = "Use web scraping to find the current weather in Philadelphia, Pennsylvania. Print the results in the terminal."
	# user_input = "Find the latest price of Ethereum online. Print the results in the terminal."
	# user_input = "Find an image of the current weather in Philadelphia, Pennsylvania. Store the image and the script you used to generate it in the locally."
	# user_input = "I want to know the current total value of the top 10 cryptocurrencies. Can you find and show each of the top ten to me in the terminal?"
	# user_input = "Get monthly unemployment rate data in the U.S. from https://data.bls.gov/timeseries/LNS14000000. Parse the table and print the data for 2022 in the terminal."
	# user_input = "Write a script that asks the user for the name of a city. Then find the population of that city, print it in the terminal, and ask the user if they want to find the population of another city. If they do, repeat the process. If they don't, end the script."
	pseudocode = generate_pseudocode(user_input)
	new_response = generate_code(user_input, pseudocode)
	errors_list=[]
	max_fix_attempts = 20
	fix_attempt = 0
	while True:
		try:
			run_code(new_response)
			# intent_met = intent_check(user_input, new_response)
			# print(intent_met) # True or False
			# if not intent_met:
			suggestions = solve_intent(user_input, new_response)
			print(suggestions)
			new_response = fix_error(old_response, suggestions, fix_attempt, max_fix_attempts)
			raise Exception("Does not seem to solve the user's intent. Try the following: " + suggestions)
		except Exception as e:
			error_msg = str(e)
			errors_list.append(error_msg)
			fix_attempt += 1
			try:
				old_response = new_response
				error_instructions = generate_error_instructions(user_input, old_response, errors_list)
				print(error_instructions,"\n\n")
				new_response = fix_error(old_response, error_instructions, fix_attempt, max_fix_attempts)
			except Exception as e:
				print(str(e))
				break
		else:
			break
	print("Attempts: ", fix_attempt)

	# Save the final response in a file with a concise, but descriptive name that limits the length of the file name
	file_name = "code_" + user_input.replace(" ", "_")[:50] + ".py"
	with open(file_name, "w") as file:
		file.write(new_response)

# Call the main function to start the process
main()