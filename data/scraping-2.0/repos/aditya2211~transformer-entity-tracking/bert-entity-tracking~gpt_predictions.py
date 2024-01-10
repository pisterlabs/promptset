import torch, json
import numpy as np
from pytorch_pretrained_bert.modeling_openai import OpenAIGPTLMHeadModel, OpenAIGPTConfig
from pytorch_pretrained_bert.tokenization_openai import OpenAIGPTTokenizer
from pytorch_pretrained_bert.optimization_openai import OpenAIAdam
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')



recipes_data = json.load(open('/scratch/cluster/agupta/recipes_elmo.json','r'))

train_data = []
val_data = []
test_data = []

for data in recipes_data:
    recipes_data[data]['para'] = []
    recipes_data[data]['targets'] = np.zeros((len(recipes_data[data]['text']),len(recipes_data[data]['ingredient_list'])))

    for step_num in range(len(recipes_data[data]['text'])):
        recipes_data[data]['para']+=recipes_data[data]['text'][str(step_num)]
    
    for step_num in recipes_data[data]['ingredients']:
        for ing in recipes_data[data]['ingredients'][step_num]:
            recipes_data[data]['targets'][int(step_num)][ing] = 1


for data in recipes_data:
    if len(recipes_data[data]['ingredient_list'])!=0 and len(recipes_data[data]['para'])!=0:
        if recipes_data[data]['split'] == 'train':
            train_data.append(recipes_data[data])
        elif recipes_data[data]['split'] == 'dev':
            val_data.append(recipes_data[data])
        else:
            test_data.append(recipes_data[data])

test_set_ing = set()
for ins in test_data:
	para_tokens = tokenizer.tokenize(" ".join(ins['para']))
	test_set_ing |= set(para_tokens)
	for ing in ins['ingredient_list']:
			test_set_ing |=  set(tokenizer.tokenize(" ".join(ing.split('_'))))



count = set()
total = set()


for ins in val_data:
	for ing in ins['ingredient_list']:
		ing_tokens = tokenizer.tokenize(" ".join(ing.split('_')))
		flag = 0
		for token in ing_tokens:
			if token not in test_set_ing:
				print('%s-%s'%(ing, token))
				count.add(ing)
		total.add(ing)


print(len(total))
print(len(count))