import transformers
import datasets
from datasets import load_dataset, load_metric, load_from_disk
import random
import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import evaluate
import nltk
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import numpy as np
from factsumm import FactSumm
import json
import openai
import sys
import os
import time
import pickle
​
​
accelerate = Accelerator()
np.random.seed(42)
max_input_length=512
max_target_length=256
openai_api_key = '<API Key>'
model_checkpoint="<Pre-trained model checkpoint>"
gpt_model_type = "gpt-3.5-turbo-0613" #gpt-4-0613
synthetic_data_type = "H2L" #L2H
​
tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
 
​
def preprocess_function(examples):
    inputs = ["Article:" + article + "\nTL;DR:" for article in examples['text']]
    model_inputs = tokenizer(inputs, max_length = max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['summary'], max_length = max_target_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs
​
def evaluate_rouge(preds, labels):
	metric = evaluate.load('rouge')
	
	preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
	labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
	
	result = metric.compute(predictions=preds, references=labels, use_stemmer=True)
	result = {key: value * 100 for key, value in result.items()}
	return {k: round(v, 4) for k, v in result.items()}
​
def get_model_summaries(model, eval_dataloader):
	result_data = {'article':[], 'reference_summary':[], 'model_summary':[], 'edited_summary':[]}
	
	device = torch.device("cuda")
	model.to(device)
	model.eval()
	
	for step,batch in enumerate(tqdm(eval_dataloader)):
		with torch.no_grad():
			generated_tokens = model.generate(batch['input_ids'].to(device),\
											 attention_mask = batch['attention_mask'].to(device),\
											 max_length=max_target_length,\
											 num_beams=4,\
											 length_penalty=0.6)
			labels = batch['labels'].numpy()
			labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
			
			decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
			decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
			decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
			
			result_data['article'].extend(decoded_inputs)
			result_data['reference_summary'].extend(decoded_labels)
			result_data['model_summary'].extend(decoded_preds)
	
	return result_data
        
def get_gpt_edited_summaries(model, prompt, data, openai_api_key, start, end, synthetic_data_type):
	openai.api_key = openai_api_key
	
	edited_summaries = []
	for i in tqdm(range(len(data['article']))):
		if (i < start):
			continue
		if (i >= end):
			break
		ex_prompt = prompt.replace("{src}", data["article"][i])
		if synthetic_data_type=="H2L":
			ex_prompt = ex_prompt.replace("{ref}", data["reference_summary"][i])
		else:
			ex_prompt = ex_prompt.replace("{ref}", data["model_summary"][i])
		received = False
		with open("./HallusinatedData/"+gpt_model_type+"-"+synthetic_data_type+".json") as json_file:
			data_saved = json.load(json_file)
		while (not received):
			try:
				response = openai.ChatCompletion.create(model=model,messages = [{'role':'user', 'content':ex_prompt}],temperature=0.1, n=1)
				received=True
			except:
				error = sys.exc_info()[0]
				if (error == openai.error.InvalidRequestError):
					assert xwFalse
				print("API error:", error)
				time.sleep(1)
		response_final = response['choices'][0]['message']['content']
		data_saved['edited_summary'].append(response_final)
		with open("./HallusinatedData/"+gpt_model_type+"-"+synthetic_data_type+".json", "w") as outfile:
			outfile.write(data_saved)
​
​
def main():
	prompt_hallucination_clinical = """
	»»»» Instruction »»»»
    
	You are a clinical writing assistant who is in edit mode. You are tasked with generating hallucinated summary based on provided a clinical note article and a reference summary for the article. The goal is to edit the reference summary to generate a hallucinated summary that sound plausible but includes edits introduced through an edit operation which can be one of the following:
    
	1. Add Operation: Intentionally add medico-legally essential words from the article not required for accurate diagnosis and treatment documentation.
	2. Omit Operation: Intentionally omit medico-legally essential words in the reference summary required for accurate diagnosis and treatment documentation.
    
    For these operations focus on words that, if missing or incorrect in the hallucinated summary, could lead to wrong diagnoses and treatments in the future. Maintain coherence while excluding essential terms. The hallucinated summary should be concise and contain no more than FIVE EXTRA WORDS compared to the reference summary and should have equal number of Add/Omit operations.
    
    Steps for generating the hallucinated summary:
    1. List the proposed edit operations to introduce hallucination on the reference summary.
    2. Use the proposed edit operations to edit the reference summary.
    
	»»»» Output Format »»»»
	The output format is: 
    	          
	Numbererd List hallucination edits made:
	{Edit 1} ||| {Edit 2} ||| {Edit 3} ...
        
	Hallucinated Summary: 
	
	»»»» Follow the above Instuctions, Hallucination Method and Output Format »»»»
    
	Now, let's start.
    
	Generate the hallucinated summary:
    
	Article - {src}
	
	Reference Summary - {ref}    
		    
	"""
	prompt_low_to_high = """
	You are a writing assistant who is in edit mode. You are tasked with generating edited summary based on provided a clinical note article and a model generated summary for the article. The goal is to edit the model generated summary to generate an edited summary that factually consistent with respect to the article and contains edits introduced through an edit operation which can be one of the following:
    
	1. Add Operation: Intentionally add medico-legally essential words (FIVE EXTRA WORDS max) from the article to the edited summary required for accurate diagnosis and treatment documentation. Only add a single sentence in a single edit.
	2. Omit Operation: Intentionally omit medico-legally non-essential words (FIVE EXTRA WORDS max) from the model generated summary to the edited summary not required for accurate diagnosis and treatment documentation. 
    
    For these operations focus on words that, if present or correct in the edited summary, could lead to right diagnoses and treatments in the future. Maintain coherence while including essential terms. The edited summary should be concise and contain no more than FIVE EXTRA WORDS compared to the model generated summary and should have equal number of Add & Omit operations.
    
    Steps for generating the edited summary:
    1. List the proposed edit operations to improve factually consistent in the model generated summary.
    2. Use the proposed edit operations to edit the model generated summary.
    3. If the edited summary does not start with "Discharge Instructions: ", then add "Discharge Instructions: " at the start of the edited summary
    
	»»»» Output Format »»»»
	The output format is: 
    	          
	Numbererd List factuality edits made:
	{Edit 1} ||| {Edit 2} ||| {Edit 3} ...
        
	Edited Summary: 
	
	»»»» Follow the above Instuctions, Factuality Improvement Method and Output Format »»»»
    
	Now, let's start.
    
	Generate the edited summary:
    
	Article - {src}
	
	Model Generated  Summary - {ref}
    
	"""
​
    
	data_files = {"train": "train.json", "valid":"valid.json", "test": "test.json"}
	dataset = load_dataset("AVL Dataset", data_files=data_files)
	dataset_train = dataset['train'].train_test_split(test_size=0.2,seed=42)
	dataset['train']=dataset_train['test']
	num_samples = 5
	train_dataset_subset = dataset['train']
	tokenized_train_dataset_subset = train_dataset_subset.map(preprocess_function, batched=True)
	tokenized_train_dataset_subset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
	batch_size = 32
	data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
	eval_dataloader = DataLoader(tokenized_train_dataset_subset, batch_size=batch_size, collate_fn=data_collator)
    
    
	result_data = get_model_summaries(model, eval_dataloader)
	result_data["reference_summary"] = train_dataset_subset['summary']
	result_data["article"] = train_dataset_subset['text']
	result_json = json.dumps(result_data, indent=4)
	with open("./HallusinatedData/"+gpt_model_type+"-"+synthetic_data_type+".json", "w") as outfile:
		outfile.write(result_json)
        
	if synthetic_data_type=="H2L":
		prompt=prompt_hallucination_clinical
	else:
		prompt=prompt_low_to_high
	get_gpt_edited_summaries(gpt_model_type, prompt, result_data, openai_api_key, 0, num_samples, synthetic_data_type)

if (__name__)=="__main__":
	main()
