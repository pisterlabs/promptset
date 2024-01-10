#Generates various types of candidate analogies
#Replace OPENAI_API_KEY with your OpenAi API key

import os
import openai
import argparse
from time import sleep

prompts_wsrc = ['Explain <target> using an analogy involving <src>.','Explain how <target> is analogous to <src>.','Explain how <target> is like <src>.','Explain how <target> is similar to <src>.','How is <target> analogous to <src>?','How is <target> like <src>?','How is <target> similar to <src>?']


prompts_nanlgy = ['Explain <target>.','What is <target>?','Explain <target> in plain language to a second grader.']

prompts_nosrc = ['Explain <target> using an analogy.','Create an analogy to explain <target>.','Using an analogy, explain <target>.','What analogy is used to explain <target>?','Use an analogy to explain <target>.']


def gen_anlgy(prompt,j,op,target,model):
	write_out = []
	if j>=0:
		for k in range(5):
			if k>=0:
				sleep(0.8)
				print(prompt)
				pred = openai.Completion.create(
				engine=model,
				prompt= prompt,
				temperature=0.85,
				max_tokens=939,
				top_p=1, 
				best_of=1,
				frequency_penalty=1.24,
				presence_penalty=1.71
				)
				print(j,k,prompt)
				print(pred)

			write_out.append(pred["choices"][0]["text"].replace('\n','').replace('\t',' ')+'\t'+target+'\t'+prompt+'\n')
			with open(op+'_p'+str(j)+'_ht.txt','a') as f:
				for r in write_out:
					f.write(r)
				write_out = []
	if j>=0:
		pred = openai.Completion.create(
		engine=model,
		  prompt= prompt,
		  temperature=0,
		  max_tokens=939,
		  top_p=1,
		  best_of=1,
		  frequency_penalty=0,
		  presence_penalty=0
		)
		print(j,prompt)
		print(pred)

		write_out.append(pred["choices"][0]["text"].replace('\n','').replace('\t',' ')+'\t'+target+'\t'+prompt+'\n')
		with open(op+'_p'+str(j)+'_lt.txt','a') as f:
				for r in write_out:
					f.write(r)
				write_out = []

def main(pt,op,model,concept_path):

	if pt == '1':
		prompts = prompts_nosrc
	elif pt == '2':
		prompts = prompts_wsrc 
	elif pt == '3':
			prompts = prompts_nanlgy
		


	with open(concept_path,'r') as f:
		data = f.readlines()

	write_out = []
	anlgies = {}
	
	for i,row in enumerate(data):
		row = row.strip('\n')
		
		row = row.split('\t')
		target = row[1].strip()
		src = row[0].strip()
		
		done_bef = False
		try:
			anlgies[target.lower()]
			done_bef = True 
		except KeyError as e:
			anlgies[target.lower()] = []
		
		print(i,target)
		if (pt == '2' or not done_bef) and i>=0:
			for j,prompt in enumerate(prompts):
				prompt = prompt.replace('<target>',target.lower())
				
				if pt == '2':
					prompt = prompt.replace('<src>',src.lower())
				gen_anlgy(prompt,j,op,target,model)
				
				

		anlgies[target.lower()].append(target)


if __name__ == '__main__':
	openai.api_key = OPENAI_KEY
	concept_path = './data/saqa_concepts.txt'
	parser = argparse.ArgumentParser(description='Enter configuration')
	parser.add_argument('--prompt_type', metavar='pt', type=str,
		help='Type of prompts. Enter 1 for analogies without source, 2 for analogies with given sources, 3 for non-analogies.',required=True)
	parser.add_argument('--outpath_prefix', metavar='op', type=str,
		help='Prefix of the output paths.',required=True)
	parser.add_argument('--model', metavar='m', type=str,
		help='Model name. Type one of the following: text-ada-001, text-babbage-001, text-curie-001, text-davinci-001.',required=True)
	

	args = parser.parse_args()
	
	if args.prompt_type not in ['1','2','3']:
		print('Please enter valid prompt type')
		exit()
	if args.model not in ['text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-001']:
		print('Please enter valid model')
		exit()
	main(args.prompt_type,args.outpath_prefix,args.model,concept_path)


	