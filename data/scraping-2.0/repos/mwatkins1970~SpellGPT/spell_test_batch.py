# This randomly selects GPT tokens from the subset of tokens which consists solely of upper and lower case roman alphabetic characters and possibly a leading space
# It iteratively runs a one-shot prompt to produce a S-P-E-L-L-I-N-G style spelling, then assesses its accuracy.
# The engine is set by default to davinci-instruct-beta, the number of runs is set by default at 100 (line 20)

import openai
import random
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel, utils, AutoTokenizer, GPTJForCausalLM


# INSERT YOUR OpenAI KEY HERE:
openai.api_key = "xx-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


engine = "davinci-instruct-beta"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
token_strings = [tokenizer.decode([i]) for i in range(50257)]

runs = 100     # Number of spellings per batch
correct = 0
tot_tok = 0
wrong_list = []

def is_latin_alpha(s):
    return bool(re.match('^[a-zA-Z]*$', s))

for i in range(runs):

	alphabetic = False

	while alphabetic == False:
		r = random.randint(0, 50256)
		targ_strg = token_strings[r].lstrip().upper()   				# target string is strg with any leading space removed and converted to upper case
		if is_latin_alpha(targ_strg) == True:
			alphabetic = True

	strg = token_strings[r]
	print(f"\n\nSTEP {i+1}")
	print(f'\nTOKEN no. {r}: "{strg}"')
	print('TARGET STRING: "' + targ_strg + '"')	

	prompt = '''If spelling the string " table" in all capital letters separated by hyphens gives\nT-A-B-L-E\nthen spelling the string "''' + strg + '''" in all capital letters, separated by hyphens, gives\n'''
	print("PROMPT: " + prompt + '\n\n')


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

	if spelling_score == 1:
		correct += 1
	else:
		wrong_list.append((strg,comp_list))

	# spelling_score should be between 0 and 1, 0.75 if ' will' gets sequentially completed as 'W', '-', 'I', '-', 'L', '-', 'K', for examples
	# This isn't a particularly good measure for 'headless tokens' like 'ureau' which spells as 'B', '-', 'U', '-', 'R', '-', 'E', '-', 'A', '-', 'U'

	print(f"\n\nTARGET STRING: {targ_strg}")
	print(f"SPELLING-OUT LIST: {comp_list}" )
	print(f"SPELLING SCORE: {spelling_score:.3f}")
	print(f"CURRENT SUCCESS RATE: {correct}/{i+1} = {correct/(i+1):.3f}")
	print(f"TOTAL TOKENS USED: {tot_tok}\n\n")

print(f"\n\nTOTAL TOKENS USED: {tot_tok}")
print(f"FINAL SUCCESS RATE: {correct}/{runs} = {correct/runs:.3f}")
print("FAILURES:")
for (strg, comp_list) in wrong_list:
	print(f"{strg}: {comp_list}")




