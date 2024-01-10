import praw
import cohere
import csv
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
import os
import time

def clean_text(text):
    negations_dic = {
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
        "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "can't": "cannot", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
        "mustn't": "must not"
    }

    combined_pat = r'|'.join((r'@[A-Za-z0-9_]+', r'https?://[^ ]+', r'www.[^ ]+'))
    www_pat = r'www.[^ ]+'
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
    tok = WordPunctTokenizer()

    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    words = [x for x in tok.tokenize(letters_only) if len(x) > 1]
    return " ".join(words).strip()

# Set up Reddit API
reddit = praw.Reddit(
    client_id='',
    client_secret='',
    user_agent='',
    username='',
    password=''
)

cohere_api_key = "YGvwkSnlrD47L5dGsVNrFmIaWz2VnwQs3f4lCVqz"
co = cohere.Client(cohere_api_key)

subreddit_name = 'legaladvice'
num_posts = 1000000000  # Adjust as needed

csv_file_path = 'posts_and_summaries.csv'

csv_mode = 'a' if os.path.exists(csv_file_path) else 'w'
with open(csv_file_path, csv_mode, newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)

    if csv_mode == 'w':
        csv_writer.writerow(['Post Body', 'Summary'])
    subreddit = reddit.subreddit(subreddit_name)
    for submission in subreddit.new(limit=num_posts):
        post_body = submission.selftext if submission.selftext else submission.url

        min_selftext_length = 256  # Adjust as needed
        if len(post_body) >= min_selftext_length:
   
            print(f"Original Post Body Length: {len(post_body)}")

            if len(clean_text(post_body)) >= min_selftext_length:
                cleaned_post_body = clean_text(post_body)
                print(f"Cleaned Post Body Length: {len(cleaned_post_body)}")

                response = co.summarize(
                    text=cleaned_post_body,
                    model='command',
                    length='short',
                    extractiveness='auto'
                )
                summary = response.summary

                csv_writer.writerow([cleaned_post_body, summary])
                print("Post written to CSV")

                time.sleep(12)
            else:
                print("Post Body too short before cleaning, skipping.")
        else:
            print("Original Post Body too short, skipping.")

print(f"Posts and summaries appended to {csv_file_path}")
