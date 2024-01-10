import glob
import json
import os
import openai
import util
import re
import sys

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

openai.api_key = os.getenv("OPENAI_API_KEY")

if len(sys.argv[1:]) != 1:
	print("Pass what dir to work on, or 'all'")

work_on = sys.argv[1:][0]


for file in glob.glob('daybook-and-diaries-1856-1906-daybook-1*/*.json'):

	print(file)
	data = json.load(open(file))

	dir = file.split('/')[-2]
	file_id = int(file.split('/')[-1].replace('.json',''))
	data['id'] = file_id
	data['dir'] = dir


	if work_on != 'all':
		if work_on not in dir:
			print('skipping',dir)
			continue


	if 'full_text' in data:


		if 'gpt' not in data:
			data['gpt'] = {}

		if 'daybook-json' in data['gpt']:
			continue

		print("WOrking on:",file)
		full_text = util.clean_up_transcribed_text(data['full_text'])

		if len(full_text) < 50:
			continue

		prompt = f"Using only the text below. Structure the following multiple diary text entries by Susan B Anthony into a valid JSON array of dictionaries extracting the date, the date again in the format yyyy-mm-dd, the city or state it was written in, other geographical locations mentioned that day, people mentioned that day, and the complete full text of the entry and a one sentence summary of the text, using the JSON keys date, dateFormated, cityOrState, geographical, people, and fullText, summaryText:\n\n---\n{full_text}\n---\n"
		print('----PROMT----')
		print(prompt)
		response = openai.Completion.create(
		  model="text-davinci-003",
		  prompt=prompt,
		  temperature=0.0,
		  max_tokens=4096 - count_tokens(prompt),
		  top_p=1,
		  frequency_penalty=0,
		  presence_penalty=0
		)
		print("response['choices'][0]")
		print(response['choices'][0])

		text_response=response['choices'][0]['text']

		print('---------text_response before=========')
		print(text_response)

		if response['choices'][0]['finish_reason'] == "length":
			print('daybook-length-too-long')
			data['gpt']['error'] = 'daybook-length-too-long'
			continue

		if text_response.find('[') == -1:
			data['gpt']['error'] = 'daybook-json'
		else:



			# trim off any extra (? why) text before the structured data
			if text_response.find('[') > 0:
				text_response = text_response[text_response.find('[')-1:].strip()	

			
			text_response = text_response.replace('\\\\',"")
			text_response = text_response.replace('\\"',"'")
			text_response = text_response.replace('("',"(")
			text_response = text_response.replace('")',")")
			text_response = text_response.replace('"-',"'")



			text_response = text_response.replace('""','')
			text_response = text_response.replace('`','')
			text_response = text_response.replace('JSON:','')
			text_response = text_response.replace('[JSON]','')


			
			text_response = text_response.replace('// etc.','')
			text_response = text_response.replace('// and so on...','')
			text_response = text_response.replace('// ...','')


			text_response = text_response.replace('"cityOrState": ,','"cityOrState": null,')
			text_response = text_response.replace('"dateFormatted": ,','"dateFormatted": null,')
			text_response = text_response.replace('"dateFormated": ,','"dateFormated": null,')
			text_response = text_response.replace('"geographical": ,','"geographical": null,')
			text_response = text_response.replace('"date": ,','"date": null,')
			text_response = text_response.replace('"people": ,','"people": null,')
			text_response = text_response.replace('"dateFormated": ,','"dateFormated": null,')
			text_response = text_response.replace('"fullText": ,','"fullText": null,')
			text_response = text_response.replace('"summaryText":\n','"summaryText": null\n')
			text_response = text_response.replace('"summaryText": \n','"summaryText": null\n')

			text_response=text_response.strip()

			# remove any trailing }, that are there
			text_response = re.sub(r'},\n\s*\n\]', '}]', text_response)			



			# find the fulltext part
			fulltext_searches = re.finditer(r'"fullText"\:(.*)', text_response, re.IGNORECASE)
			fulltext_searches_findall = re.findall(r'"fullText"\:(.*)', text_response, re.IGNORECASE)
			print('-------text_response after replace cleanup')
			print(text_response)
			print('------')
			print("fulltext:")
			print("fulltext_searches",list(fulltext_searches))
			print('fulltext_searches_findall',fulltext_searches_findall)

			if len(list(fulltext_searches_findall)) == 0:
				print("fulltext_search failed, try on next run")
				continue

			for fulltext_search in fulltext_searches_findall:
				print('fulltext_search',fulltext_search)

				if fulltext_search.strip() == 'null,':
					continue

				replace_with = fulltext_search
				replace_with = replace_with.replace("\\",'')
				replace_with = replace_with.replace('"','')
				replace_with = replace_with.replace('\ ','')


				replace_with = f'"{replace_with}",'

				print("Replacing with:",replace_with)

				text_response = text_response.replace(fulltext_search,replace_with)

			print('------text_response post fulltext regex')
			print(text_response)
			# # pull out the geographical key and parse it, if it fails nuke it, its too complicated to parse why a json array could be malformed
			# geo_search = re.search(r'"geographical"\:(.*)', text_response, re.IGNORECASE)

			# geo_text = geo_search.group(1).strip()
			# if geo_text[-1] == ',':
			# 	geo_text = geo_text[0:-1].strip()

			
			# try:
			# 	json.loads(geo_text)
			# except:
			# 	print(text_response)
			# 	print("geographical parse failed, setting it to empty")
			# 	text_response = text_response.replace(geo_search.group(1),'[],')



			print('---------')
			print(text_response)		
			print('---------')
			jsonResponse = json.loads(text_response)			
			data['gpt']['daybook-json'] = jsonResponse

			
	
	json.dump(data,open(file,'w'),indent=2)