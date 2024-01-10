# %%
import os
import openai
import time 
from tqdm import tqdm, trange
import pandas as pd
from copy import deepcopy
from collections import Counter
import json 

openai.api_key = os.environ['OPENAI_API_KEY']
assert os.environ['OPENAI_API_KEY'] != ""

current_folder = os.path.dirname(__file__)

df_tax = pd.read_json(os.path.join(
	current_folder,
	'..',
	'..',
	'data',
	'taxonomy',
	'wish_newtax.json'
), lines=True)

paths = set(df_tax[df_tax['is_leaf']]['category_path'].str.lower())

# change this
infer_file = os.path.join(
	current_folder,
	'..',
	'..',
	'data',
	'wish_products',
	'wish-mturk-labelled-09202022-clean-joinedlance.json'
)

# change this
out_path = os.path.join(
	current_folder,
	'..',
	'..',
	'data',
	'wish_products',
	'wish-mturk-labelled-09202022-clean-joinedlance-openai_reverse_inferred.json'
)

df_labelled_path = os.path.join(
	current_folder,
	'..',
	'..',
	'data',
	'wish_products_truetag_tahoe',
	'Wish_Truetag_Tahoe_Meta_Train_OpenaiReverseCreated.json'
)

# %%
df = pd.read_json(df_labelled_path, lines=True)
df['category'] = df['category'].apply(lambda x: " > ".join(x))
p = set(df['category'])
assert p == paths

df_infer = pd.read_json(infer_file, lines=True)

#%%
with open(out_path, 'a', buffering=1) as f:
	for i in tqdm(df_infer.to_dict('records')):
		try:
			prompt = None
			tmp_recs = []
			for _ in trange(200):
				if prompt is None:
					prompt = df.sample(40).drop_duplicates('category')
				prompt = '\n'.join((prompt['title'].str.strip() + ' -> ' + \
					prompt['category']).str.strip().tolist())
				prompt = prompt + '\n' + i['title'].strip() + " -> "
				response = openai.Completion.create(
					model="text-davinci-002",
					prompt=prompt,
					temperature=0,
					max_tokens=256,
					top_p=1,
					frequency_penalty=0,
					presence_penalty=0,
					stop=["\n"]
				)
				tax = response['choices'][0]['text'].strip()
				if tax in paths:
					i['openai_reverse_predicted_category'] = tax.split(" > ")
					tmp_recs.append(deepcopy(i))
					prompt = df.sample(40).drop_duplicates('category')
				else:
					# progressive refinement
					if len([j for j in paths if tax in j]) > 0:
						tmp = df[df.category.apply(lambda x: tax in x)]
						prompt = tmp.sample(min(40, len(tmp)))
					else:
						level = -1
						while True:
							tax_tmp = " > ".join(tax.split(" > ")[:level])
							if len(tax_tmp) == 0:
								break
							if len([j for j in paths if tax_tmp in j]) > 0:
								tmp = df[df.category.apply(lambda x: tax_tmp in x)]
								prompt = tmp.sample(min(40, len(tmp)))
								break
							level -= 1
				if len(tmp_recs) == 10:
					break
			
			i['openai_reverse_predicted_categories'] = Counter(
				[tuple(k['openai_reverse_predicted_category']) for k in tmp_recs]).most_common()
			del i['openai_reverse_predicted_category']
			f.write(json.dumps(i) + '\n')
			f.flush()
		except Exception as e:
			print(i, e)
			time.sleep(10)