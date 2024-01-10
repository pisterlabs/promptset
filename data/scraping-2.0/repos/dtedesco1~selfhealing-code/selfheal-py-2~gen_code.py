from config import openai, model, general_prompt_boilerplate

code_prompt_boilerplate = "DO NOT explain what you're doing in natural language. DO NOT use a codeblock. ONLY respond with code."

# Define the function to generate python code based on a prompt
def code(prompt):
	# Send the user input to the model to generate a script
	response = openai.ChatCompletion.create(
		model=model,
		messages=[
			{"role": "system", "content": "You are a bot that responds to prompts with only working, errorless Python code and nothing else. Your outputs can be pasted directly into a Python file and run."},
			{"role": "user", "content": "Write Python code that meets the following user need:\nPrint the word 'foobar'\nYour code:"},
			{"role": "assistant", "content": "print('foobar')"},
			{"role": "user", "content": "Perfect! Thank you. Let's try another."},
			{"role": "user", "content": "Write Python code that meets the following user need:\nPrint the numbers 1 through 10'\nYour code:"},
			{"role": "assistant", "content": "print('def print_numbers():\n    for i in range(1, 11):\n        print(i)')"},
			{"role": "user", "content": "Perfect! Thank you. Let's try another."},
			{"role": "user", "content": prompt,}],
		max_tokens=3000,
		n=1,
		stop=None,
		temperature=0.7
	)
	# Extract the generated script from the API response
	response = response['choices'][0]['message']['content']
	return response

# Function to generate the code prompt and generate the code.
def code_output(input, suggestions):
	prompt = f"Write Python code that meets the following user need:\n{input}\n\nPlease follow these suggestions:\n{suggestions}\n{general_prompt_boilerplate}\n{code_prompt_boilerplate}\n\nYour code output:"
	output = code(prompt)
	with open("output.py", "w") as f:
		f.write(output)
	return output

# Function to generate the tests prompt and generate the tests.
def code_tests(input, output, prev_tests='No tests written yet', test_results='No tests run yet'):
	# Simplified prompt excluding the previous tests and test results
	prompt = f"Write tests for the following code (stored in the local directory as 'output.py'):\n{output}. Previous tests failed in this way: {test_results}\n{code_prompt_boilerplate}\n\nYour new test code:"

	# generate a prompt asking the AI to write tests for the code to ensure it meets the user's needs
	# prompt = f"Write tests for the following code (stored in the local directory as 'output.py'):\n{output}\n\nYour previous tests (stored as 'tests.py'):\n{prev_tests}\nThe code should meet the following user need:\n{input}\n\nYour previous code failed with the following error(s):\n{test_results}\n\n{general_prompt_boilerplate}\n{code_prompt_boilerplate}\n\nYour new test code:"
	tests = code(prompt)
	with open("tests.py", "w") as f:
		f.write(tests)
	return tests