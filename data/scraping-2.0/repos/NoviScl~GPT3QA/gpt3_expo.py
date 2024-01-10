import enum
import os
import openai
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from evaluate import * 
import numpy as np
np.random.seed(2021)

openai.api_key = ''
openai.Engine.retrieve('davinci-msft')

def load_data(train_path, test_path):
	'''
	load train and test data
	'''
	with open(train_path, 'r') as f:
		train = json.load(f)

	with open(test_path, 'r') as f:
		test = json.load(f)
	
	return train, test
	
def select_demo(data, num_demo=4):
	'''
	Select k demo examples from the training data.
	'''
	# random sample
	return np.random.choice(data, num_demo)


def qa_prompt(train, num_demo=4):
	'''
	Construct the prompt for QA task.
	'''
	selected_demos = select_demo(train, num_demo)
	prompt = ''
	for i in range(num_demo):
		prompt += selected_demos[i]["question"] + '\n'
		answer = selected_demos[i]["answer"]
		prompt += "The answer is " + answer + "\n\n"
	return prompt


def predict(question, num_demo=4):
	prompt = qa_prompt()
	input_prompt = prompt + question + '\n'
	input_prompt += "The answer is"
	aa = openai.Completion.create(
		engine=engine,
		prompt=input_prompt,
		temperature=0.0,
		max_tokens=8,
		top_p=1.0,
		frequency_penalty=0.0,
		presence_penalty=0.0,
		logprobs=1,
		stop=['\n']
	)
	ans = aa['choices'][0]['text']
	confidence = sum(aa['choices'][0]["logprobs"]["token_logprobs"])
	## some simple cleaning
	ans = ans.split()
	ans_lst = []
	for a in ans:
		ans_lst.append(a)
		if a[-1] == '.':
			break
	ans = ' '.join(ans)

	# print ("question: ", question)
	# print ("prediction: ", ans)
	# print ("confidence: ", confidence)
	# print ("length: ", len(question.split()))

	# if (confidence > -10.0 and len(question.split()) > 100) or (confidence > -6.0 and question.split() > 60):
	# 	return ans, confidence
	# else:
	# 	return None
	if len(question.split()) > threshold:
		return ans
	else:
		return None


def quizbowl_test(samples=100, threshold=100):
	answers = []
	test = test_100[ : samples]
	EM = 0
	total_len = 0
	buzz_len = 0
	buzzed = 0
	for eg in test:
		total_len += len(eg["text"].split())
		text_orig = sent_tokenize(eg["text"])
		for s in range(len(text_orig)):
			question = ' '.join(text_orig[:s+1])
			pred = predict(question, threshold=threshold)
			
			if pred is not None:
				buzz_len += len(question.split())
				buzzed += 1

				## clean up gold answer
				ans = eg["answer"].split('[')[0].strip()
				ans = ans.split('(')[0].strip()
				# print ("answer: ", eg["answer"])
				# print ("\n")

				# print ("pred: ", pred)
				# print ("gold: ", ans)
				# print ()

				em = get_exact_match(ans, pred)
				EM += em

				break
	print ("EM score: ", EM)
	print ("AVG question len: ", total_len / 100)
	print ("buzzed: ", buzzed)
	print ("AVG len of buzz: ", buzz_len / buzzed)
	print ("\n")
	return answers



if __name__ == "__main__":
	# for threshold in [90, 100, 110]:
	# 	print ("buzz threshold: ", threshold)
	# 	answers = quizbowl_test(100, threshold=threshold)
	train, test = load_data('DiverseQA/NQ_train.json', 'DiverseQA/NQ_test.json')


