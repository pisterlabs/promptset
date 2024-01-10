'''
For classified non-analogies from the analoginess scorer, this script refines the prompt by appending the non-analogy with the original prompt.
Modify analogies_path/out_path as needed.
Replace OPENAI_API_KEY with your OpenAI API key.
'''
import openai
from time import sleep
def prompter(prompt,tmp):
	resps = []
	if tmp == 'ht':
		for j in range(5):
			resp = openai.Completion.create(
		  engine="text-davinci-001",
		  prompt= prompt,
		  temperature=0.85,
		  max_tokens=939,
		  top_p=1, 
		  best_of=1,
		  frequency_penalty=1.24,
		  presence_penalty=1.71
		)
			print(resp)
			resps.append(resp["choices"][0]["text"].strip('\n').replace('\n',''))
		
	else:
		resp = openai.Completion.create(
				  engine="text-davinci-001",
				  prompt= prompt,
				  temperature=0,
				  max_tokens=939,
				  top_p=1,
				  best_of=1,
				  frequency_penalty=0,
				  presence_penalty=0
				)
		resps.append(resp["choices"][0]["text"].strip('\n').replace('\n',''))
		print(resp)	
	return resps


def read_f(path,src=False):
	lines = []
	targets = []
	prompts = []
	tmps = []
	domains = []
	pred_cls = []

	with open(path) as f:
		for row in f.readlines():
			splt = row.strip('\n').split('\t')
			lines.append(splt[0])
			targets.append(splt[1])
			prompts.append(splt[2])
			tmps.append(splt[3])
			domains.append(splt[4])
			pred_cls.append(splt[5])

	return lines,targets,prompts,tmps,domains,pred_cls

def main(analogies_path,out_path):
	analogies, targets, prompts, tmps, domains, pred_cls = read_f(analogies_path)
	with open(out_path,'a') as f:
		for idx,analogy in enumerate(analogies):
			
			if idx>=0:
				
				if pred_cls[idx] == '0':
					print(idx,analogy)
					
					sleep(0.3)
					new_anlgies = []
					prompt = analogy + '\n' + prompts[idx]
					new_anlgies = prompter(prompt,tmps[idx])
					for na in new_anlgies:
						f.write(na+'\t'+targets[idx]+'\t'+prompts[idx]+'\t'+tmps[idx]+'\t'+domains[idx]+'\t'+analogy+'\n')
									

if __name__ == '__main__':
	analogies_path = '../data/analoginess_scorer/non_adapt.txt'
	out_path = '../data/prompt_refiner/non_adapt.txt'
	openai.api_key = OPENAI_API_KEY
	main(analogies_path,out_path)
