# This inputs a string (ideally a GPT token) and iteratively runs a one-shot prompt to produce a S-P-E-L-L-I-N-G style spelling, then assesses its accuracy.

import openai
import re

# INSERT YOUR OpenAI KEY HERE:
openai.api_key = "xx-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


engine = "davinci-instruct-beta"



# Input a string (supposedly a token, but this will accept anything made of spaces and Roman alphabetic characters)

alphabetic = False

while alphabetic == False:
	strg = input("Please enter a (token) string: ")  	
	targ_strg = strg.lstrip().upper()   				# target string is strg with any leading space removed and converted to upper case
	if targ_strg.isalpha() == True:
		alphabetic = True

print('\n\nINPUT STRING: "' + strg + '"')
print('TARGET STRING: "' + targ_strg + '"')	

prompt = '''If spelling the string " table" in all capital letters separated by hyphens gives\nT-A-B-L-E\nthen spelling the string "''' + strg + '''" in all capital letters, separated by hyphens, gives\n'''
print("PROMPT: " + prompt + '\n\n')



tot_tok = 0

sofarsogood = True      # spelling-is-correct-thus-far Boolean
pos = 0         		# relevant position in the target string
comp_list = []   		# this should be a list of tokens that the model attempted to spell the string with 

# tokens to ignore when assessing correct spelling
fillers = [' ', '\n', '~', '-', '--', '---', '----', '-----', '------', '-------', '--------', '---------', '----------', '-----------', '-----------', '------------', '-------------', '--------------', '---------------' '–', '—']

EOT = False		 		# "end-of-text has been reached" Boolean variable

while pos < len(targ_strg) and EOT == False:			# work our way through the target string one character at a time
	target = targ_strg[pos]       # this is the character in the target string we're checking for
	comp = ' '					  # arbitarily chosen filler string so that the next 'while' holds true

	while comp in fillers: 				# keep going if you have hyphen-type characters, blank spaces or line breaks
		comp_len = 0                   # first, produce ever longer completions until you get one that's not all line breaks (this had been a problem)
		nonbreak = False				# not-a-linebreak Boolean

		# This is a bit messy, but stops long strings of line breaks clogging things up
		while comp_len < 50 and nonbreak == False:      # keep going until we have a non-linebreak token (or 50 tokens)
			comp_len += 1
			response = openai.Completion.create(engine=engine, temperature = 0, prompt = prompt, max_tokens = comp_len)
			comp = response["choices"][0]["text"]
			tot_tok += response['usage']['total_tokens']

			comp = comp.replace('\n','')    
			if comp != '':
				nonbreak = True        #If the completion comp was all linebreaks, it's now == '', so comp != '' means it wasn't all linebreaks
			if comp == '<|endoftext|>':
				EOT = True
						# What happens if you get > 50 linebreaks? The comp comes back as ''. I don't recall seeing that except at the end of a spelling.
						# Note how on each iteration, the prompt allow more and more tokens in the completion. So it keeps extending the lenght of the rollout
						# until there's a non-linebreak character or 50 linebreaks. There's probably a better way to do this!


		prompt += comp              # append completion to prompt to send back through

		print(prompt)

	comp_list.append(comp)		# add completion to list (should be a token, and a single upper case letter in most cases) 

	if comp.upper().replace('-','').replace('–','').replace('—','') != target and sofarsogood == True:
		sofarsogood = False 
		spelling_score = pos/len(targ_strg)
	pos += 1

if sofarsogood == True:
	spelling_score = 1

# spelling_score should be between 0 and 1, 0.75 if ' will' gets sequentially completed as 'W', '-', 'I', '-', 'L', '-', 'K', for examples
# This isn't a particularly good measure for 'headless tokens' like 'ureau' which spells as 'B', '-', 'U', '-', 'R', '-', 'E', '-', 'A', '-', 'U'

print(f"\n\nTARGET STRING: {targ_strg}")
print(f"SPELLING TOKEN LIST: {comp_list}" )
print(f"SPELLING SCORE: {spelling_score:.3f}")
print(f"TOKENS USED: {tot_tok}\n\n")
