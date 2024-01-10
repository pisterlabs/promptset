"""
This script will be used to send prompts to the 
GPT API and post the replies to reddit.
"""

# %% imports

import os
import openai
from dotenv import load_dotenv
import get_reddit_posts as grp

# %%

# store API credntials
load_dotenv()
openai.api_key = os.environ.get('OPEN_AI_KEY')
openai.Model.list()

def find_relevant_posts(submissions_df, num_posts=1):
	"""
	Find posts that seem good to respond to.
	You can adjust your criteria to choose what you like.
	This could be based on score, keywords, etc.
	"""

	top_post = submissions_df.sort_values(by='created_utc', ascending=False).head(num_posts)

	return top_post

def prompt_gpt(prompt: str): 
	"""
	Take a string and send it to GPT as a prompt
	"""
	completion = openai.ChatCompletion.create(
	model="gpt-3.5-turbo",
	messages=[{"role": "user", "content": prompt}])

	return completion.choices[0].message

def has_ai_language(response: str) -> bool:
	"""
	This function will check if the response contains
	the "As an AI language model..." disclaimer.
	
	"""

	check_string = "As an AI language model".lower()

	if check_string in response.lower():
		return True
	else:
		return False

# check ai language function
def test_has_ai_language():
	assert True == has_ai_language("As an AI language model, I don't have personal opinions.")
	assert False == has_ai_language("I don't have personal opinions.")



def check_post_history(post_id: str, history_file: str) -> bool:
	"""
	This function will update the post history with
	the post id of the post that was responded to.

	If the post id is already in the file, it will
	return True. Otherwise it will write the post id
	to the file and return False.
	"""
	
	# check for post id in file
	# https://stackoverflow.com/questions/15233340
	with open(history_file, 'r') as f:
		post_ids = f.read().splitlines()

	print(post_ids)
	# check if post id is in file
	if post_id in post_ids:
		print("Post has already been responded to.")
		return True
	
	# add post id to file
	with open(history_file, 'a') as f:
		f.write(post_id + '\n')
	
	return False

# test the check_post_history function
def test_check_post_history():
	assert True == check_post_history('test1', 'data/post_history.txt')


# %% main 

if __name__ == '__main__':

		print('Running GPT Bot')

		reddit = grp.reddit_connection()

		# get posts
		submissions = reddit.subreddit('AskReddit').new(limit=10)

		# this function is a simpler version of what we had earlier
		submissions_df = grp.create_submission_df(submissions)

		top_posts = find_relevant_posts(submissions_df, 10)

		# check if there are any posts to respond to
		for index, submission in top_posts.iterrows():
			print(submission['id'])

			# check if post has been responded to
			if check_post_history(submission['id'], 'data/post_history.txt'):
				# skip this post
				continue

			# sleep for 20 seconds to prevent rate limiting
			import time
			time.sleep(20)

			prompt = submission['title']

			print("Post title:", prompt)

			# prompt GPT
			response = prompt_gpt(prompt)

			print("GPT response:", response.content)

			# check if response is AI language
			if has_ai_language(response.content):
				print("Response is AI language")
				continue # skip this post

			# post response to reddit
			reply = reddit.submission(submission['id']).reply(response.content)

			print('posted: ', reply)

		# get posts history from our bot

		my_comments_list = reddit.user.me().comments.top('all')

		comments_df = grp.create_comments_df(my_comments_list)

		import datetime

		filename = 'bot_history_' + datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S-%p")

		comments_df.to_csv('data/' + filename + '.csv')

# %%
