import openai
import os
import praw
from praw.models import MoreComments

# poetry add praw
# pip install praw

openai.api_key = os.getenv("OPENAI_API_KEY")
reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")

reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent="sentiment_test",
            username="codemke")

# finanance hot sorting
for submission in reddit.subreddit("finance").hot(limit=5):

    # extrac title and id
    print(f"title: {submission.title}({submission.id})")

    # get submission
    submission = reddit.submission(submission.id)

    title = f"title: {submission.title}"

    # extract comments
    for cnt, top_level_comment in enumerate(submission.comments):
        if isinstance(top_level_comment, MoreComments):
            continue

        print(f"comment: {top_level_comment.body}")

        comment = f"comment: {top_level_comment.body}"
        prompt = f"extract the company name from the title, and classify sentiment as positive or neutral or negative for the following comment.\
            give the short answer in format as company name and following positive, neutral or negative only\n\n{title}\n{comment}"

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
        )
        print(response['choices'][0]['text'])

        if cnt > 5:
            break
    
    print("---------")
