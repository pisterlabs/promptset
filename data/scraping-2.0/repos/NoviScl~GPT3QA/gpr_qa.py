import os
import openai
import json
from tqdm import tqdm
from evaluate import * 
import numpy as np
from tfidf_retriever import *
np.random.seed(2021)

engine = 'davinci'
openai.api_key = ''
openai.Engine.retrieve(engine)

SAVE_DIR = 'retriever_caches'
MODEL_PATH = 'tfidf.pickle'
INDEX_PATH = 'index.pickle'
QN_PATH = 'questions.pickle'
ANS_PATH = 'answers.pickle'

def load_data(train_path, test_path):
	'''
	load train and test data
	'''
	with open(train_path, 'r') as f:
		train = json.load(f)

	with open(test_path, 'r') as f:
		test = json.load(f)
	
	return train, test
	
def select_demo_random(data, num_demo=4):
	'''
	Select k demo examples from the training data.
	'''
	# random sample
	return np.random.choice(data, num_demo)




def qa_prompt(train, num_demo=4):
	'''
	Construct the prompt for QA task.
	'''
	selected_demos = select_demo_random(train, num_demo)
	prompt = ''
	for i in range(num_demo):
		prompt += selected_demos[i]["question"] + '\n'
		answer = selected_demos[i]["answer"][0]
		prompt += "The answer is " + answer + "\n"
	return prompt



def retrieve_prompt(question, QA_data="NQ", num_demo=4):
	'''
	Select k demo examples from the training data.
	data: 
	QA_data: specifies the dataset.
	'''
	tfidf_guesser = TfidfGuesser()
	tfidf_guesser.load(QA_data)	
	top_q, top_a = tfidf_guesser.guess(question = question, max_n_guesses = num_demo)
	prompt = ''
	for i in range(num_demo):
		prompt += top_q[i] + '\n'
		answer = top_a[i]
		prompt += "The answer is " + answer + "\n"
	return prompt


def predict(question, QA_data="NQ", num_demo=4):
	prompt = retrieve_prompt(question, QA_data=QA_data, num_demo=num_demo)
	input_prompt = prompt + question + '\n'
	input_prompt += "The answer is"
	aa = openai.Completion.create(
		engine=engine,
		prompt=input_prompt,
		temperature=0.0,
		max_tokens=6,
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

	return ans, confidence



if __name__ == "__main__":
	QA_data = "QANTA"
	guesstrain = os.path.join("DiverseQA", QA_data + "_train.json")
	guesstest = os.path.join("DiverseQA", QA_data + "_test.json")
    
	train, test = load_data(guesstrain, guesstest)
	# for qd in test:
	# 	question = qd["question"]
	# 	prompt = retrieve_prompt(question, QA_data, num_demo=4)
	# 	print (question)
	# 	print (prompt)
	# 	break
    
	predictions = []
	em = 0
	for qd in tqdm(test):
		pred = {}
		question = qd["question"]
		pred["question"] = question
		gold_ans = qd["answer"]
		pred["gold_answer"] = gold_ans
		pred_ans, conf = predict(question, QA_data=QA_data, num_demo=16)
		pred["prediction"] = pred_ans
		pred["confidence"] = conf
		predictions.append(pred)

		em += get_exact_match(pred_ans, gold_ans)

	print ("EM: ", em / len(test))
	with open(os.path.join("predictions", QA_data+"_test_PromptRetrieve_preds.json"), "w") as f:
		json.dump(predictions, f)





