from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
tokenizer = GPT2Tokenizer.from_pretrained('/Users/yingying/Desktop/pre_train_model/gpt2')
model = GPT2LMHeadModel.from_pretrained('/Users/yingying/Desktop/pre_train_model/gpt2')
model.eval()

# PromptTemplate
"""
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["prompt_key", "year_day"],
    template="请到系统查询一下截止今天我的{prompt_key},{year_day}?",
)
print(prompt.format(prompt_key="剩余调休" year_day="年假信息"))
"""
# text = "Based on the following known information, provide concise and professional answers to user questions. " \
#        "Adding fabricated elements to the answer is not allowed.Known content:This short sleeve is suitable for people weighing 115-170." \
#        "Question:Can people weighing 167 wear this short sleeve?"

text = "PageRank is an algorithm that measures the importance of webpages based on the links pointing to them. " \
       "The basic idea is that authoritative pages get more links. So pages with more links should rank higher in search results. " \
       "Especially if those links come from popular pages (i.e., pages that have high PageRank scores themselves)." \
       "Previously, SEOs could see the PageRank score of any webpage via the Google Toolbar." \
       "Answer the question based on known information:what is pagerank?"
# text = "the color of the clothes is Black." \
#        "the sizes of the clothes: 0-1, 2-3, 9-10." \
#        "the target audience for clothing is Women." \
#        "the material of the clothes is Silk."\
#        "let's think step by step:  what is the color of the clothes?"
encoded_input = tokenizer.encode(text)
answer_len = len(encoded_input)
tokens_tensor = torch.tensor([encoded_input])

stopids = tokenizer.convert_tokens_to_ids(["."])[0]
past = None
for i in range(100):
    with torch.no_grad():
        # gpt2的参数："n_embd": 768, "n_head": 12, "n_layer": 12, "vocab_size": 50257
        # "n_ctx"(所能允许的文本最大长度): 1024, "n_positions"(通常与 n_positions 相同): 1024
        output, past = model(tokens_tensor, past_key_values=past, return_dict=False)
        # past_key_values 保存了上次迭代过程中的 key 和 value（attention 运算中的键值对）用于加速运算
        # past_key_values: ((K, Q)) * 12,
        # 因此past_key_values 的结构为 (12，2，(batch_size, num_head, sql_len, head_features))
        # 即 (12，2，(1, 12, sql_len, 64)) --> sql_len 为 encoded_input 的 length

    token = torch.argmax(output[..., -1, :])

    encoded_input += [token.tolist()]

    if stopids == token.tolist():
        break
    tokens_tensor = token.unsqueeze(0)

sequence = tokenizer.decode(encoded_input[answer_len:])

print(sequence)
