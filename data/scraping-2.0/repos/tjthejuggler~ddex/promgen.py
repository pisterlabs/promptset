import json
import openai
import random
from os import path
import argparse
import re
import os
import textwrap

#if the user runs promgen.py with the argument (python3 promgen.py -d) ((for detailed description))
#	then we want to use our ai to make a more descriptive prompt then what the user gave
#	python3 promgen.py "hatice is very>" --------  hatice is very good at being an amazing person
#   python3 promgen.py "hatice ran fast" --------  



#right now if the user does -e or -o then they need to follow this format (-e 3,4,5) (-o 2,3,4),
#	make it so they can use numbers or the words, for example (-e religious,realistic) = (-e 2,4)



cwd = os.getcwd() #

use_CD_format = True

api_key = os.environ.get("OPEN_AI_API")
openai.api_key = api_key 

engine = "ada"
engine = "davinci"

category_keys = {
			'2': 'religious', '3': 'hyperrealistic', '4': 'realistic', '5': 'surreal',
			'6': 'abstract', '7': 'fantasy', '8': 'cute', '9': 'people', '10': 'creatures', '11': 'nature',
			'12': 'buildings', '13': 'space', '14': 'objects', '15': 'boats', '16': 'cars',
			'17': 'pencil',	'18': 'paint', '19': 'CGI', '20': 'colorful', '21': 'dull', '22': 'black and white',
			'26': 'new','27': 'old','28': 'creepy',	'29': 'cartoon'
			}

def load_modifiers():
	styles_file = open( "promgen_styles.txt", "r")
	styles = styles_file.readlines()
	styles_file.close()
	artists_file = open( "promgen_artists.txt", "r")
	artists = artists_file.readlines()
	artists_file.close()
	artists_dict = {}
	#artists_dict = {'bob' : ["religious", "hyperrealistic"], "charlie" : ["hyperrealistic", "happy"]}
	if path.exists('./promgen_artists_formatted.txt'):	
		print('file exists')		
		with open('./promgen_artists_formatted.txt') as json_file:
			artists_dict = json.load(json_file)
	keywords_file = open( "promgen_keywords.txt", "r")
	keywords = keywords_file.readlines()
	keywords_file.close()
	prompts_file = open( "promgen_prompts.txt", "r")
	pre_prompts = prompts_file.readlines()
	prompts_file.close()
	artist_intros = ["in the style of","by","inspired by","resembling"]
	return (styles, artists_dict, keywords, pre_prompts, artist_intros)

def get_args():
	user_input, batch_size , use_detail_description = 'a boy', 2, False
	every_categories_filter, only_categories_filter = [], []
	parser = argparse.ArgumentParser(	    
		prog='PROG',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		description=textwrap.dedent('''\
			for use with -e and -a, comma seperated (ex: -e 1,5,14)
			Category Key:
			2 religious 
			3 hyperrealistic (very close to photos) - this one may be rare
			4 realistic (actual real things, but obviously not a photo)
			5 surreal (breaks usual physics)
			6 abstract (lots of shapes and patterns, not everything is identifiable)
			7 fantasy (witches, angels, magic, dragons, faries..)
			8 cute
			9 people
			10 creatures (real or unreal animals)
			11 nature
			12 buildings
			13 space
			14 objects
			15 boats 
			16 cars
			17 pencil
			18 paint
			19 CGI
			20 colorful
			21 dull (not bright colors)
			22 black and white
			26 new
			27 old
			28 creepy (scary evil scary big animals)
			29 cartoon
			'''))

	parser.add_argument("prompt", help="the base prompt (comma seperate each weighted section")
	parser.add_argument("-b", "--batchsize", type = int, help="batch_size, the number of images")
	parser.add_argument("-e", "--everycat", type = str, help="use every modifier in these categories")
	parser.add_argument("-o", "--onlycat", type = str, help="use only modifiers that have all these categories")
	parser.add_argument("-d", "--details", action = "store_true", help="ai makes detail description")

	args = parser.parse_args()
	if args.batchsize:
		batch_size = args.batchsize	
	if args.everycat:
		every_categories_filter = [x for x in args.everycat.split(",")]
	if args.onlycat:
		only_categories_filter = [x for x in args.onlycat.split(",")]
	if args.details:
		use_detail_description = True
	user_input = args.prompt
	return (user_input, batch_size, every_categories_filter, only_categories_filter, use_detail_description)

def rand_item(my_list, is_artist):
	intro = ''
	if is_artist:
		intro = 'by '
	return intro+random.choice(my_list).strip()

def rand_w():
	to_ret = str(random.randint(1,9))
	return (to_ret)

def get_gpt_result(user_prompt, pre_prompts):
	prompt = """
	* """+ random.choice(pre_prompts) + """
	* """+ random.choice(pre_prompts) + """
	* """+ random.choice(pre_prompts) + """
	* """+ random.choice(pre_prompts) + """
	* """+ random.choice(pre_prompts) + """
	* """+ user_prompt
	response = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=30, stop= "\n")
	result = response["choices"][0]["text"].strip()
	result = result.replace(',',(":"+rand_w()+'", "')).replace('.',(":"+rand_w()+'", "'))
	return result

def get_task_result(user_prompt):
	prompt = '''make these sentences very interesting and descriptive, but only use one sentence.\n
	a man is running - a man running like the wind, his feet barely touching the ground.\n
	'''+ user_prompt +' - '
	response = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=30, stop= "\n")
	result = response["choices"][0]["text"].strip()
	result = result.replace(',',(":"+rand_w()+'", "')).replace('.',(":"+rand_w()+'", "'))
	return result

def create_output_file(filename, output_lines):
	folder_name = engine
	with open(cwd+'/'+folder_name+'/'+filename + '.txt', 'w') as f:
		for item in output_lines:
			f.write("[%s]\n" % item.strip("\n"))

def get_every_filter(artists_dict, filter_list):
	listOfKeys = list()
	listOfItems = artists_dict.items()
	for filter_key in filter_list:
		filter_word = category_keys[filter_key]
		for artist in artists_dict.keys():
			if filter_word in artists_dict[artist] and artist not in listOfKeys:
				listOfKeys.append(artist)
	return listOfKeys

def get_only_filter(artists_dict, filter_list):
	listOfKeys = list()
	listOfItems = artists_dict.items()
	for artist in artists_dict.keys():
		should_append = True
		for filter_key in filter_list: 
			filter_word = category_keys[filter_key]
			if not (filter_word in artists_dict[artist] and artist not in listOfKeys):
				should_append = False
		if should_append:
			listOfKeys.append(artist)
	return listOfKeys

def main():
	user_input, batch_size, every_categories_filter, only_categories_filter, use_detail_description = get_args()
	styles_file = open( "promgen_styles.txt", "r")
	styles, artists_dict, keywords, pre_prompts, artist_intros = load_modifiers()
	artist_intros = ["in the style of","by","inspired by","resembling"]
	filtered_artists = list(artists_dict.keys())
	if every_categories_filter and only_categories_filter:
		print("You can't use 'every' filter and 'only' filter together")
		quit()
	elif every_categories_filter:
		filtered_artists = get_every_filter(artists_dict, every_categories_filter)
		#print(filtered_artists)
	elif only_categories_filter:
		filtered_artists = get_only_filter(artists_dict, only_categories_filter)
		#print(filtered_artists)
	user_prompt = ''	
	prompts = []
	#print('user_input', user_input) #a big dog,@
	for i in range(batch_size):#this will run once for each prompt it will create
		prompt_to_append = ''
		for section in user_input.split(","): #analyze 
			section = section.replace('"', '')
			#print('section', section) # a big red dog  -------------- @

			if len(prompt_to_append) > 1: #if we have already been through once, then make a ,
				prompt_to_append = prompt_to_append + ","
			prompt_to_append = prompt_to_append + '"'
			if section[0] == "$": #style is used
				prompt_to_append = prompt_to_append + rand_item(styles, False)
			elif section[0] == "@": #artist is used
				prompt_to_append = prompt_to_append + rand_item(filtered_artists, True)
			elif section[0] == "^": #keyword is used
				prompt_to_append = prompt_to_append + rand_item(keywords, False)
			elif section[0] == ":":
				if section[-1] == ":": #if the char after the : is not a digit, then
					prompt_to_append = prompt_to_append + rand_item(random.choice([artists,styles,keywords]))+":"+rand_w()+'"'
			else:
				if ">" in section and section[0] != ">":
					user_prompt = section.split(">")[0]
					result = ""
					if use_detail_description: 
						result = get_task_result(user_prompt)
						prompt_to_append = prompt_to_append + ' ' + result
					else:
						result = get_gpt_result(user_prompt, pre_prompts)
						prompt_to_append = prompt_to_append + user_prompt+' '+result
					if section[-1] == ":":
						prompt_to_append = prompt_to_append + ":"+rand_w()
					prompt_to_append = prompt_to_append + ":"+rand_w()+'"'
				else:
					if ":" in section:
						prompt_to_append = prompt_to_append + section+'"'
						if section[-1] == ":":
							prompt_to_append = prompt_to_append + rand_w()+'"'
					else:
						prompt_to_append = '"'+ section + '"'
			if section[0] == "$" or section[0] == "@" or section[0] == "^":
				if len(section) > 1: #    @:4  
					if section[1] == ":" and section[-1] == ":": #if no weight is given, then use a random weight
						prompt_to_append = prompt_to_append + ":"+rand_w()
					else:
						prompt_to_append = prompt_to_append + ":" + section.split(":")[1]
				prompt_to_append = prompt_to_append + '"'
		print('prompt_to_append', prompt_to_append)
		prompt_to_append = prompt_to_append.replace('""', '"').replace('" ', '"')
		prompt_to_append = re.sub(' +', ' ', prompt_to_append).replace(" ,]", "]")

		prompt_to_append = re.sub('":\d+','', prompt_to_append)
		prompt_to_append = re.sub('", :\d+','", ', prompt_to_append)
		prompt_to_append = prompt_to_append.replace('", ","', ", ")
		#:5:8
		prompt_to_append = re.sub(':\d:\d',':'+rand_w(), prompt_to_append)
		prompt_to_append = prompt_to_append.replace('", ", ', '')
		prompt_to_append = prompt_to_append.replace(':"', ':')
		prompts.append(prompt_to_append)
	create_output_file(user_prompt.replace(' ', '_')+str(random.randint(0,1000000)), prompts)
	for item in prompts:
		if item.endswith(', "'):
			item = item.split(', "')[0]
		print("["+item.replace('""','"')+"],")

main()

