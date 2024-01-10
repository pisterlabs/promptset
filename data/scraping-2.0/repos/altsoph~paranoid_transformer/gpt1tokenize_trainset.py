import random
import nltk
from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer

model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

SPECIAL_TOKENS = ["<long>", "<quotes>", "<others>", "<cyberpunk>", "<text>", "<meta1>", "<meta2>"]

# We can add these special tokens to the vocabulary and the embeddings of the model:
tokenizer.set_special_tokens(SPECIAL_TOKENS)
model.set_num_special_tokens(len(SPECIAL_TOKENS))

MAX_LEN = 500

dataset = []
for fn,meta1,meta2 in (('long_cyberpunk.txt','<long>','<cyberpunk>'),('quotes_cyberpunk.txt','<quotes>','<cyberpunk>'),
					   ('long_others.txt','<long>','<others>'),('quotes_others.txt','<quotes>','<others>')):
	meta_tokens = tokenizer.convert_tokens_to_ids((meta1,meta2))
	for line in open(fn, encoding='utf-8', errors='ignore'):
		if not line.strip(): continue
		# meta_tokens = tokenizer.encode("%s %s" %(meta1,meta2))
		# segments = tokenizer.convert_tokens_to_ids(segments)
		tokens = tokenizer.encode(line.strip())
		if len(tokens)>MAX_LEN:
			# print('too long',len(tokens))
			sentences = nltk.sent_tokenize(line.strip())
			# print(sentences)
			sentences_tokens = [tokenizer.encode(sentence) for sentence in sentences]
			# print(sentences_tokens)
			collected = []
			for sentence_tokens in sentences_tokens:
				if 0 in sentences_tokens or len(collected)+len(sentence_tokens)>MAX_LEN:
					# print(len(collected),collected)
					dataset.append( (meta1,meta2,meta_tokens+collected) )
					collected = []
				if len(sentence_tokens)<=MAX_LEN:
					collected.extend(sentence_tokens)
			if collected:
				# print(len(collected),collected)
				dataset.append( (meta1,meta2,meta_tokens+collected) )
			# exit()
		else:
			dataset.append( (meta1,meta2,meta_tokens+tokens) )
for m1,m2,token_ids in dataset:
	print("%s\t%s\t%s" % (m1,m2,",".join(map(str,token_ids))))
