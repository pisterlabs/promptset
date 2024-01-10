import openai
from dotenv import load_dotenv
import os
import praw

# load environment variables from .env file
load_dotenv()

########### Define helper functions ########

# Define a function to get title and comments from Reddit
def get_titles_and_comments(subreddit="stocks", limit=6, num_comments=3, skip_first=2):
    subreddit = reddit.subreddit(subreddit)
    titles_and_comments = {}

    counter = 0
    for counter, post in enumerate(subreddit.hot(limit=limit)):

        if counter >= skip_first:
            index = counter - skip_first
            titles_and_comments[index] = ""

            submission = reddit.submission(id=post.id)
            title = post.title
            titles_and_comments[index] += "Title: " + title + "\n\n"
            titles_and_comments[index] += "Comments: " + submission.selftext + "\n\n"

            comment_counter = 0
            for top_level_comment in submission.comments:
                if comment_counter < num_comments:
                    if top_level_comment != '[deleted]':
                        titles_and_comments[index] += top_level_comment.body + "\n"
                        comment_counter += 1
                if comment_counter == num_comments:
                    break # break out of comments loop

        counter += 1

    return titles_and_comments

def print_titles_and_comments(titles_and_comments):
    for key, value in titles_and_comments.items():
        print(key, value)
        print("")

def create_prompt(titles_and_comments):
    task = """
    Return the stock ticker or company name mentioned in the title and comments.
    Classify the sentiment around the company  as positive, negative, or neutral.
    If no ticker or company is mentioned, return '[No company mentioned]'.
    """
    prompt = task + "\n\n" + titles_and_comments
    return prompt

############ Authenticate ###########

# Authenticate with Reddit
reddit = praw.Reddit(client_id=os.environ["REDDIT_CLIENT_ID"],
                     client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                     user_agent=os.environ["REDDIT_USER_AGENT"])


# Authenticate with OpenAI                             
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

########### Main program ###########
for key, title_with_comments in get_titles_and_comments().items():
    prompt = create_prompt(title_with_comments)
    
    response = openai.Completion.create(engine="text-davinci-003",
                                    prompt=prompt,
                                    max_tokens=256,
                                    temperature=0.1,
                                    top_p=1.0)
    print(title_with_comments)
    print(f"Sentiment Report from OpenAI: {response.choices[0].text}")
    print("---------------------------------")

