#%%
import os
import openai
from tqdm import tqdm
import json
import pandas as pd
import random
import time
#%%
openai.api_key = os.environ['OPENAI_API_KEY']
assert os.environ['OPENAI_API_KEY'] != ""

current_folder = os.path.dirname(__file__)
df_truetag_prompts = os.path.join(
	current_folder,
	'..',
	'..',
	'data',
	'wish_products_truetag_tahoe',
	'wishproducts_truetag_tahoe_stratsample.json'
)

df_reverse_generated_newtag_products = os.path.join( 
	current_folder,
	'..',
	'..',
	'data',
	'wish_products_truetag_tahoe',
	'Wish_Truetag_Tahoe_Meta_Train_OpenaiReverseCreated.json'
)

df_out_path = os.path.join(
	current_folder,
	'..',
	'..',
	'data',
	'wish_products_truetag_tahoe',
	'Wish_Truetag_Tahoe_Meta_Train_OpenaiReverseCreated_AddedInferredTrueTag.json'
)

df_truetag_prompts = pd.read_json(df_truetag_prompts, lines=True)
df_products_reverse_generated = pd.read_json(df_reverse_generated_newtag_products, lines=True)

#%%
recs = []
for i in df_truetag_prompts.to_dict('records'):
	try:
		idx = random.sample(range(len(i['categories'])), 1)[0]
		recs.append({
			'title': i['title'],
			'category_truetag': i['categories'][idx],
			'text': i['title'] + ' -> ' + ''.join(['[' + j + ']' for j in i['categories'][idx]])
		})
	except:
		pass
df_truetag_prompts = pd.DataFrame(recs)

#%%
with open(df_out_path, 'w', buffering=1) as fout:
	for rec in tqdm(df_products_reverse_generated.to_dict('records')):
		try:
			title = rec['title'].strip()
			response = openai.Completion.create(
				model="text-davinci-002",
				prompt='\n'.join(df_truetag_prompts.sample(30)['text'].tolist()) + '\n' + title + " ->",
				temperature=0,
				max_tokens=256,
				top_p=1,
				frequency_penalty=0,
				presence_penalty=0,
				stop=["\n"]
			)

			tax_truetag = response['choices'][0]['text'].strip()
			rec['openai_truetag_category'] = tax_truetag[1:-1].split('][')
			fout.write(json.dumps(rec))
			fout.write('\n')
		except Exception as e:
			print(e)
			time.sleep(10)