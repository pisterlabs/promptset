import os
import openai
import time
import json

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv('OPENAI_API_KEY')
dirname = os.path.dirname(__file__)

def get_completion(prompt, model="gpt-3.5-turbo"):

	messages = [{"role": "user", "content": prompt}]
	response = openai.ChatCompletion.create(
			model=model,
			messages=messages,
			temperature=0.5,
	)

	return response.choices[0].message["content"]

def format_data(ref, response, verse):
	data = {}
	data['book'] = str(ref.split(" ")[0])
	data['chapter'] = int(ref.split(" ")[1].split(":")[0])
	data['start_verse'] = int(ref.split(" ")[1].split(":")[1])
	data['end_verse'] = int(ref.split(" ")[1].split(":")[1])
	data['context'] = str(verse)
	data['questions'] = json.loads(response)

	return data

def get_prompt(verse):

	prompt = f'''
		Your task is to generate question answer pairs given a Bible verse as context.
		These question answer pairs are meant to be simple, imagine that they are for a 2nd grade comprehension quiz.

		Here are some examples:

		verse: "God saw that the light was good, so God separated the light from the darkness."
		response:
		[
			{{
				"question": "What did God separate the light from?",
				"answer": "The darkness"
			}},
			{{
				"question": "What did God separate from the darkness?",
				"answer": "The light"
			}}
		]

		verse: "God said, "Let the water under the sky be gathered to one place and let dry ground appear." It was so."
		response: 
		[
			{{
				"question": "What did God want to be gathered to one place?",
				"answer": "The water"
			}},
			{{
				"question": "What appeared when the water was gathered to one place?",
				"answer": "Dry ground"
			}}
		]

		Do not create any duplicate questions.
		You can create anywhere between 1 - 4 questions depending on the amount of content in the verse. 
		The answers should be nouns and as short as possible.
		Make sure you use the exact format as seen above.

		verse: "{verse}"
		response: 
	'''

	return prompt

def main():
	filename = os.path.join(dirname, 'vref.txt')
	refs_file = open(filename, encoding="utf-8")
	refs = refs_file.readlines()

	filename = os.path.join(dirname, 'en-NET.txt')
	verses_file = open(filename, encoding="utf-8")
	verses = verses_file.readlines()

	# Book of Ruth
	start_index = 7129
	end_index = 7214

	data = []

	try:

		for i, verse in enumerate(verses[start_index:end_index]):
			prompt = get_prompt(verse.strip())
			response = get_completion(prompt)
			formatted = format_data(refs[start_index+i].strip(), response, verse)
			data.append(formatted)

			print(f"Line {start_index+i} Completed")
			time.sleep(20)
	
	except Exception as e:
		print(e)
	
	finally:
		output_file = open(f'{dirname}/output.json', "w")
		json.dump(data, output_file, ensure_ascii=False)
			
	return

main()