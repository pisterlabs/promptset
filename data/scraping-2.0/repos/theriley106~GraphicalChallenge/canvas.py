import sys
import time
import json
import threading
import pandas as pd
import os
import pickle
import re
import os
import openai
from openai.embeddings_utils import distances_from_embeddings
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
try:
	from keys import *
except:
	API_KEY = os.environ['OPEN_AI']

openai.api_key = API_KEY

def split_into_many(text, max_tokens, tokenizer):

	# Split the text into sentences
	sentences = text.split('. ')

	# Get the number of tokens for each sentence
	n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
	
	chunks = []
	tokens_so_far = 0
	chunk = []

	# Loop through the sentences and tokens joined together in a tuple
	for sentence, token in zip(sentences, n_tokens):

		# If the number of tokens so far plus the number of tokens in the current sentence is greater 
		# than the max number of tokens, then add the chunk to the list of chunks and reset
		# the chunk and tokens so far
		if tokens_so_far + token > max_tokens:
			chunks.append(". ".join(chunk) + ".")
			chunk = []
			tokens_so_far = 0

		# If the number of tokens in the current sentence is greater than the max number of 
		# tokens, go to the next sentence
		if token > max_tokens:
			continue

		# Otherwise, add the sentence to the chunk and add the number of tokens to the total
		chunk.append(sentence)
		tokens_so_far += token + 1

	return chunks

def create_context(
	question, df, max_len=2000, size="ada"
):
	"""
	Create a context for a question by finding the most similar context from the dataframe
	"""

	# Get the embeddings for the question
	q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

	# Get the distances from the embeddings
	df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


	returns = []
	cur_len = 0

	# Sort by distance and add the text to the context until the context is too long
	for i, row in df.sort_values('distances', ascending=True).iterrows():
		print(row)
		
		# Add the length of the text to the current length
		cur_len += row['n_tokens'] + 4
		
		# If the context is too long, break
		if cur_len > max_len:
			break
		
		# Else add it to the text that is being returned
		returns.append(row["text"])

	# Return the context
	return "\n\n###\n\n".join(returns)

def answer_question(
		df,
		model="text-davinci-003",
		question="Am I allowed to publish model outputs to Twitter, without a human review?",
		max_len=2000,
		size="ada",
		debug=False,
		max_tokens=150,
		stop_sequence=None
	):
		"""
		Answer a question based on the most similar context from the dataframe texts
		"""
		context = create_context(
			question,
			df,
			max_len=max_len,
			size=size,
		)
		# If debug, print the raw model response
		# if debug:
		print("Context:\n" + context)
		# print("\n\n")

		try:
			# Create a completions using the questin and context
			response = openai.Completion.create(
				prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, search externally to gather information and combine it with the contxt. \n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
				temperature=0,
				max_tokens=max_tokens,
				top_p=1,
				frequency_penalty=0,
				presence_penalty=0,
				stop=stop_sequence,
				model=model,
			)
			return response["choices"][0]["text"].strip()
		except Exception as e:
			print(e)
			return ""

def fetch_df_for_course(course_id):
    return pd.read_pickle("{}.pkl".format(course_id))
	
# input("AYY")


max_tokens = 2000

if __name__ == "__main__":
	if len(sys.argv) == 2:
		COURSEWORKS = "https://courseworks2.columbia.edu"
	else:
		COURSEWORKS = sys.argv[2]
	COURSE = sys.argv[1]


	

	df = fetch_df_for_course(COURSE)

	

	while True:

		print(answer_question(df, question=input("Question: "), debug=False))