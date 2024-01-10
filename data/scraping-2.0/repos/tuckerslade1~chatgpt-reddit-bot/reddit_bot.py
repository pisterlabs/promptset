from dotenv import load_dotenv
import os
import praw
import openai
import datetime
import time
import pytz
import random

# todo: logging comments, respond to being corrected, add models for more subreddits, move script to cloud to run 24/7, add conversation following and replying to replies, delete comments that get downvoted, document process on personal website

load_dotenv()
eastern = pytz.timezone('US/Eastern')
openai.api_key = os.getenv('OPENAI_API_KEY')

subs = {
    'explainlikeimfive' : {
        'model' : os.getenv('EXPLAINLIKEIMFIVE_MODEL'),
        'system_message' : 'You are a friendly, knowledgeable Reddit user explaining a concept in simple terms. Please do not include hyperlinks or edit messages in your reply, or provide details about your life as if you are a real person.',
    },
    'relationship_advice' : {
        'model' : os.getenv('RELATIONSHIPADVICE_MODEL'), 
        'system_message' : 'You are a friendly Reddit user giving relationship advice. Please do not include hyperlinks or edit messages in your reply, or provide details about your life as if you are a real person.',
    },
    'changemyview' : {
        'model' : os.getenv('CHANGEMYVIEW_MODEL'),
        'system_message' : 'You are a friendly, knowledgeable Reddit user trying to change someone\'s view. Please do not include hyperlinks or edit messages in your reply, or provide details about your life as if you are a real person.',
    },
    'writingprompts' : {
        'model' : os.getenv('WRITINGPROMPTS_MODEL'),
        'system_message' : 'You are a creative Reddit user writing a story. Please do not include hyperlinks or edit messages in your reply, or provide details about your life as if you are a real person.',
    },
    'gaming' : {
        'model' : os.getenv('GAMING_MODEL'),
        'system_message' : 'You are a friendly reddit user who enjoys playing video games and is knowledgeable about them. Please do not include hyperlinks or edit messages in your reply, or provide details about your life as if you are a real person.',
    }
}

with open('replied_posts.txt', 'a+') as f:
    f.seek(0)  # Move file pointer to the beginning of the file
    replied_posts = set(line.strip() for line in f)  # Read post IDs into a set

with open('logs.txt', 'a+') as g:
    g.seek(0)  # Move file pointer to the beginning of the file
    logs = set(line.strip() for line in g)  # Read post IDs into a set

def main():

    # api credentials
    reddit = praw.Reddit(
        client_id = os.getenv('REPLYER_CLIENT_ID'),
        client_secret = os.getenv('REPLYER_CLIENT_SECRET'),
        user_agent = os.getenv('REPLYER_USER_AGENT'),
        username = os.getenv('REPLYER_REDDIT_USERNAME'),
        password = os.getenv('REPLYER_REDDIT_PASSWORD'),
    )
    current_sub = list(subs.keys())[0]
    subreddit = reddit.subreddit(current_sub)

    # listen for new posts and reply
    while True:
        current_time = datetime.datetime.now(eastern).strftime("%c")
        log(f"{current_time}:   Sorting posts in {current_sub} by hot...")
        min_post_score = 30
        max_post_comments = 40
        min_post_age = 60*4 # seconds
        max_post_age = 60*60*24 # seconds
        for submission in subreddit.hot(limit=100):
            posted = False
            current_time = datetime.datetime.now(eastern).strftime("%c")
            log(f"{current_time}:   Checking post: {submission.id} - Subreddit: {current_sub} Score: {submission.score} - Comments: {submission.num_comments} - Age: {datetime.datetime.now().timestamp() - submission.created_utc}")
            if submission.id not in replied_posts and submission.score >= min_post_score and submission.num_comments <= max_post_comments and datetime.datetime.now().timestamp() - submission.created_utc >= min_post_age and datetime.datetime.now().timestamp() - submission.created_utc <= max_post_age:
                process_submission(submission, current_sub)
                posted = True
                break
            if posted:
                break

        # try sorting by new if no suitable posts found in hot
        current_time = datetime.datetime.now(eastern).strftime("%c")
        log(f"{current_time}:   No suitable posts found in hot. Sorting by new...")
        min_post_score = 5
        max_post_comments = 15
        min_post_age = 60*4 # seconds
        max_post_age = 60*60*24 # seconds
        for submission in subreddit.new(limit=100):
            current_time = datetime.datetime.now(eastern).strftime("%c")
            log(f"{current_time}:   Checking post: {submission.id} - Subreddit: {current_sub} - Score: {submission.score} - Comments: {submission.num_comments} - Age: {datetime.datetime.now().timestamp() - submission.created_utc}")
            # trying to select posts that are interesting, relatively new and not flooded with replies
            if submission.id not in replied_posts and submission.score >= min_post_score and submission.num_comments <= max_post_comments and datetime.datetime.now().timestamp() - submission.created_utc >= min_post_age and datetime.datetime.now().timestamp() - submission.created_utc <= max_post_age:
                process_submission(submission, current_sub)
                break

        current_sub = list(subs.keys())[0]
        subreddit = reddit.subreddit(current_sub)
        sleep_time = random.randint(180, 361)
        log(f"{current_time}:   Next sub: {current_sub} - Sleeping for {sleep_time} seconds...\n")
        time.sleep(sleep_time)


def process_submission(submission, current_sub):

    prompt = submission.title + '\n\n' + submission.selftext
    model = subs[current_sub]['model']
    system_message = subs[current_sub]['system_message']
    reply_text = generate_reply(prompt, model, system_message)
    current_time = datetime.datetime.now(eastern).strftime("%c")
    if reply_text:
        log(f"{current_time}:   Replying to:   {submission.id} - {submission.title}\n")
        log(f"{current_time}:  \"{reply_text}\"\n")
        submission.reply(reply_text)
        replied_posts.add(submission.id)
        with open('replied_posts.txt', 'a') as f:  # Append the post ID to the file
            f.write(f"{submission.id}\n")


def generate_reply(prompt, model_id, system_instruction=None, max_tokens=512, max_attempts=3, min_length=80):
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    assistant_reply = ''
    for _ in range(max_attempts):
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            n=1,
            stop=['edit:','Edit:','EDIT:'],
            temperature=0.9,
        )

        assistant_reply = response['choices'][0]['message']['content'].strip()

        # check if the reply is complete by looking for end of sentence marker (only somewhat accurate)
        if assistant_reply[-1] in ['.', '?', '!'] and len(assistant_reply) >= min_length:
            return assistant_reply
    current_time = datetime.datetime.now(eastern).strftime("%c")
    if len(assistant_reply) < min_length:
        log('Failed to generate reply of minimum length.')
    elif assistant_reply[-1] not in ['.', '?', '!']:
        log(f'{current_time}:   Error: Failed to generate reply within token limit.')
    else:
        log(f'{current_time}:   Error: Failed to generate reply for unknown reason. Last attempt: \n\"{assistant_reply}\"')
    return False

def log(message):
    print(message)
    with open('logs.txt', 'a') as f:  # Append the post ID to the file
        f.write(f'{message}\n')


if __name__ == '__main__':
    main()