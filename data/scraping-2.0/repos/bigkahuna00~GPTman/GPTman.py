import os
import sys
import openai

openai.api_key = ""

base_prompt = "You are a language model designed to provide optimized documentation for Unix command-line utilities, libraries, system calls and more. Your goal is to help users better understand the purpose, function, and usage of these utilities by providing detailed and user-friendly documentation. You should avoid technical jargon and use natural language wherever possible to make the documentation more accessible. Please organize your documentation by topic and function and provide specific examples of uses and use cases where appropriate to help users better understand how to use the utilities."

model = "gpt-3.5-turbo"

#Check if the API key is set
if not openai.api_key:
	if os.getenv("OPENAI_API_KEY") == None:
		print("You must have an OpenAI API key to use this script. Please set one as an environment variable or include it in the 'openai_api_key' variable of the script.")
		sys.exit()
	else:
		openai.api_key = os.getenv("OPENAI_API_KEY")	

def GPTman(user_prompt):

	final_prompt = base_prompt + user_prompt
		
	query = openai.ChatCompletion.create(model=model, messages=[{"role":"system", "content":base_prompt}, {"role":"user", "content":user_prompt}])
	
	print(query.choices[0].message.content.strip())

def main():
	if len(sys.argv) < 2:
		print(f"""Usage: python3 {sys.argv[0]} <query>

Description: 
This script uses the ChatGPT to generate optimized documentation for Unix command-line utilities, libraries, system calls and more.

Arguments:
query: A string containing the user prompt. This prompt should describe the command or utility for which you need documentation.

Example Usage: 
To get documentation for the 'grep' utility, run: python3 {sys.argv[0]} grep

Note: 
You must have an OpenAI API key to use this script.""")

	else:
		GPTman(sys.argv[1])

if __name__ == "__main__":
	main()
