from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import torch

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
model = OpenAIGPTModel.from_pretrained("openai-gpt")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

print('inputs[input_ids]', inputs['input_ids'])
print('inputs[attention_mask]', inputs['attention_mask'])
print('outputs.last_hidden_state.shape: ', outputs.last_hidden_state.shape)