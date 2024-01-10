# %%
import os
import openai
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='Train|Val|Test')
args = parser.parse_args()

openai.api_key = os.environ['OPENAI_API_KEY']
assert os.environ['OPENAI_API_KEY'] != ""


mode = args.mode
assert mode in ['Train', 'Val', 'Test']

current_folder = os.path.dirname(__file__)
df_tax_path = os.path.join(
		current_folder,
		'..',
		'..',
		'data',
		'taxonomy',
		'wish_newtax.json'
)

df_wish_tahoe_path = os.path.join(
		current_folder,
		'..',
		'..',
		'data',
		'wish_products_truetag_tahoe',
		f'Wish_Truetag_Tahoe_Meta_{mode}.json'
)

df_wish_tahoe_out_path = os.path.join(
		current_folder,
		'..',
		'..',
		'data',
		'wish_products_truetag_tahoe',
		f'Wish_Truetag_Tahoe_Meta_{mode}_OpenaiReverseCreated.json'
)

#%%
import pandas as pd
df_tax = pd.read_json(df_tax_path, lines=True)
df_tax = df_tax[(df_tax.category_path.apply(len) > 0) & (df_tax.is_leaf)]
category_path_prompts = sorted(list(set(df_tax.category_path.str.lower().apply(lambda x: '>'.join(x.split(' > '))))))
df_tahoe_chunks = pd.read_json(df_wish_tahoe_path, lines=True, chunksize=30)

from tqdm import trange, tqdm
import time
import json
# %%

while True:
	df_tahoe = next(df_tahoe_chunks)
	prompt_demo = (df_tahoe.category.apply(lambda x: '>'.join(x)) + ': ' + '"' + df_tahoe.title + '"').tolist()
	if len(df_tahoe):
		with open(df_wish_tahoe_out_path, 'a', buffering=1) as fout:
			for i in tqdm(category_path_prompts):
				try:
					np.random.shuffle(prompt_demo)
					prompt_todo = prompt_demo + \
						["Imitate above examples' semantic styles, write accurate and creative product title given category below:"] + \
						[i + ': ' + '"']
					tries = 10
					while tries > 0:
						tries -= 1
						response = openai.Completion.create(
							model="text-davinci-002",
							prompt='\n'.join(prompt_todo),
							temperature=1,
							max_tokens=256,
							top_p=1,
							frequency_penalty=0,
							presence_penalty=0,
							stop=["\n"]
						)
						title = response["choices"][0]["text"][:-1]
						if len(title):
							break
					if len(title):
						rec = {
							"title": title, 
							"category": i.split('>'),
							"text": title + ' -> ' + ''.join(['[' + j + ']' for j in i.split('>')])
						}
						fout.write(json.dumps(rec))
						fout.write('\n')
					else:
						print(f"No product generated for {i}")
				except Exception as e:
					print(e)
					time.sleep(1)
		os.system(f"dvc add {df_wish_tahoe_out_path}")
		time.sleep(10)
		print(f'{df_wish_tahoe_out_path} added to dvc')
