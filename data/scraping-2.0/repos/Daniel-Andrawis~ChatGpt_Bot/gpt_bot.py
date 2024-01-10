# %% imports
import os
import openai
from dotenv import load_dotenv
import reddit_posts as rp

# %%

# store API credentials
load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
openai.Model.list()

def find_relevant_posts(submissions_df, num_posts=1):
    """
    This function finds the most relevant posts in the 
    subreddit for the GPT bot to respond to.
    """
    top_post = submissions_df.sort_values(by='created_utc', ascending=False).head(num_posts)
    return top_post

def prompt_gpt(prompt: str): 
    """
    This function takes a string and sends it to GPT as a prompt.
    """
    completion = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5)

    return completion.choices[0].text

def has_ai_language(response: str) -> bool:
    """
    This function checks if the response contains the "As an AI language model..." disclaimer.
    """
    check_string = "As an AI language model".lower()
    if check_string in response.lower():
        return True
    else:
        return False

def check_post_history(post_id: str, history_file: str) -> bool:
    """
    This function updates the post history with the post ID of the post that was responded to.
    If the post ID is already in the file, it returns True. Otherwise, it writes the post ID to the file and returns False.
    """
    with open(history_file, 'r') as f:
        post_ids = f.read().splitlines()

    # check if post ID is in file
    if post_id in post_ids:
        print("Post has already been responded to.")
        return True
    
    # add post ID to file
    with open(history_file, 'a') as f:
        f.write(post_id + '\n')
    
    return False

# %% main 

if __name__ == '__main__':
    print('Running GPT Bot')

    reddit = rp.reddit_connection()

    # get posts
    submissions = reddit.subreddit('AskReddit').new(limit=10)

    # create dataframe
    submissions_df = rp.create_submission_df(submissions)

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

        print("GPT response:", response)

        # check if response is AI language
        if has_ai_language(response):
            # post the response
            submission.reply(response)
            print("Posted response to Reddit.")

            # update post history
            check_post_history(submission['id'], 'data/post_history.txt')
# %%
