if __name__ == "__main__":
	from config import API_KEY
else :
	from .config import API_KEY

import openai
import streamlit as st
import tiktoken

openai.api_key = API_KEY



@st.cache_data
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
	"""Returns the number of tokens used by a list of messages."""
	try:
		encoding = tiktoken.encoding_for_model(model)
	except KeyError:
		encoding = tiktoken.get_encoding("cl100k_base")
	if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
		num_tokens = 0
		for message in messages:
			num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
			for key, value in message.items():
				num_tokens += len(encoding.encode(value))
				if key == "name":  # if there's a name, the role is omitted
					num_tokens += -1  # role is always required and always 1 token
		num_tokens += 2  # every reply is primed with <im_start>assistant
		return num_tokens
	else:
		raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def generate_messages(system_prompt, prompt, tweets) :
	messages = []
	if system_prompt :
		messages.append({"role": "assistant", "content": system_prompt})
	messages.append({"role": "assistant", "content": prompt+tweets})
	return messages

def get_n_tweets(n, tweets) :
	splited = tweets.split("\n\n")
	if len(splited) <= n :
		return tweets
	else :
		return "\n\n".join(splited[:n])

def messages_of_max_length(system_prompt, prompt, tweets, max_num_tokens=2048) :
	nb_tweets = 1
	first_tweets = get_n_tweets(nb_tweets, tweets)
	messages = generate_messages(system_prompt, prompt, first_tweets)
	num_tokens = num_tokens_from_messages(messages)
	
	while num_tokens < max_num_tokens :

		if nb_tweets == len(tweets.split("\n\n")) :
			break
		
		nb_tweets += 1
		first_tweets = get_n_tweets(nb_tweets, tweets)
		messages = generate_messages(system_prompt, prompt, first_tweets)
		num_tokens = num_tokens_from_messages(messages)

	if nb_tweets == len(tweets.split("\n\n")) :
		return messages
	else :
		nb_tweets -= 1
		first_tweets = get_n_tweets(nb_tweets, tweets)
		messages = generate_messages(system_prompt, prompt, first_tweets)
		return messages

@st.cache_data
def summarize(sentences) :
	# system_prompt = "You are a reporter trying to summarize a series of tweets about an event. They are given to you by the user and your goal is to summarize them in a way that is coherent and easy to read but also includes the most important information whithout being too long."
    
	# prompt = "Give a very short summary (less than a 100 words) of the following tweets describing an incident : \n"
	prompt = "Describe the event happening in the following tweets as if you were writing a wikipedia introduction. Be concise (less than 100 words) :\n"
	tweets = "\n\n".join(sentences)

	messages = messages_of_max_length("", prompt, tweets)

	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo-0301",
		messages=messages,
		temperature=0,
	)

	return response['choices'][0]['message']['content']

if __name__ == "__main__":
	system_prompt = "You are a reporter trying to summarize a series of tweets about an event. They are given to you by the user and your goal is to summarize them in a way that is coherent and easy to read but also includes the most important information whithout being too long."
    
	prompt = "Summarize the following tweets in less than a 150 words:\n"
	messages = generate_messages(system_prompt, prompt, "")
	print(f"Number of tokens in the system prompt and the prompt : {num_tokens_from_messages(messages)}")

	example_tweets = [ # tweets about trump
		"1 RT @TheOnion: Trump: I’m Not Going To Let The Fake News Media Tell Me What To Do https://t.co/6Z7ZQ5Z7ZQ",
		"2 RT @realDonaldTrump: The Fake News Media is going crazy with their conspiracy theories and blind hatred. @FoxNews is better!",
		"3 CNN is going crazy with their conspiracy theories and blind hatred. @FoxNews is better!",
		"4 Fake News Media is going crazy with their conspiracy theories and blind hatred. @FoxNews is better!",
	]


	tweets = "\n\n".join(example_tweets)

	print(f"Number of tokens in the system prompt, the prompt and the tweets : {num_tokens_from_messages(generate_messages(system_prompt, prompt, tweets))}")

	messages = messages_of_max_length(system_prompt, prompt, tweets)

	print(f"Number of tokens in the system prompt, the prompt and the reduced tweets : {num_tokens_from_messages(messages)}")

	# print(messages)
	encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
	encoded = encoding.encode(example_tweets[0])
	decoded = encoding.decode(encoded)
	print(f"Example tweet : {example_tweets[0]}")
	print(f"Encoded tweet : {encoded}")
	print(f"Decoded tweet : {decoded}")

	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo-0301",
		messages=messages,
		temperature=0,
	)


