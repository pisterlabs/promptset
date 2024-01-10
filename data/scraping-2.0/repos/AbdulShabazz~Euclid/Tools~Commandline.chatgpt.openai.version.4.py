import os
import re
import sys
import json
import time as time
import openai
import threading

# OPENAI
# @param - prompt [prompt.length + response.length < max_tokens]
# @param - model [gpt-3.5-turbo/gpt-4(13pg max/gpt-4-32k(52pg max))] or [[code/text]-[davinci/curie/babbage/ada]-[ver]]
# @param - max_tokens [prompt + response < 4097] Length. Bandwidth alloted for the session/connection 
# @param - temperature [1.0 - 0.0] Strictness. Controls the creativity or randomness of the response
# @param - top_p [1.0 - 0.0] Predictability. Controls the constrained randomness of oncomming text
# @param - frequency_penalty [1.0 - 0.0] Readability. Controls the ratio of used uncommon words
# @param - presence_penalty [1.0 - 0.0] DisAllowed repetitiveness/Use synonyms. Ratio of input tokens allowed in the response
# @returns  - { choices[{ engine:[davinci, curie], finish_reason:[stop,length], index:N, logprob:[], text:[response]},] }
def issueQuery(myprompt,g):
	openai.api_key = os.getenv("OPENAI_API_KEY") # Set before calling this module
	response = openai.Completion.create(model="text-davinci-003", max_tokens=4097, presence_penalty=0.0, top_p=1.0, temperature = 1.0, prompt=json.dumps(myprompt))
	for choice in response.choices:
		#result = word + " = " + choice.text
		g.writelines(choice.text.lower() + "\n")
		#print(choice.text + "\n")

MAX_TOKENS = 4097 # total syllables = prompt + completions
MAX_SEND_TOKENS = 425
MAX_THREADS = 18
SINGLE_THREADED = False

def main():

	alltasks = []
	worker = []

	if len(sys.argv) < 3:
		print("Usage: python script.py <prompt> <file>")
		sys.exit(1)

	myprompt = sys.argv[1]
	fn = sys.argv[2]	

	with open(fn,"r") as f:
		worker += [myprompt + f", `{f.read()}`"]

	with open("ChatGPT.log","w+") as g:
		for prompt in worker:
			try:
				if SINGLE_THREADED:
					issueQuery(prompt)
				else:
					time.sleep(0.001)
					th = threading.Thread(target=issueQuery, args=(prompt,g))
					th.start()
					alltasks += [th]
				worker = []
			except Exception as e:
				print(e)
				#time.sleep(1)
		if len(worker):
			issueQuery(worker,g)
			worker = []
		if len(alltasks):
			for th in alltasks:
				th.join()

print("Q.E.D.")

if __name__ == "__main__":
	main()