import openai
import json
from transformers import BertJapaneseTokenizer
from transformers import BertModel

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')


import json

# 入力用の文章をロード
with open('./docs.json') as f:
    docs = json.load(f)

index = []
for doc in docs:
    input_s = tokenizer(doc['title'], return_tensors="pt")
    outputs = bert_model(**input_s)
    last_hidden_states = outputs.last_hidden_state
    
    attention_mask = input_s.attention_mask.unsqueeze(-1)
    valid_token_num = attention_mask.sum(1)
    
    base_vec = (last_hidden_states*attention_mask).sum(1) / valid_token_num
    base_vec = base_vec.detach().cpu().numpy()[0]
    # ベクトルをデータベースに追加
    index.append({
        'title': doc['title'],
        'embedding': base_vec.tolist()
    })

with open('./index.json', 'w') as f:
    json.dump(index, f,ensure_ascii=False)