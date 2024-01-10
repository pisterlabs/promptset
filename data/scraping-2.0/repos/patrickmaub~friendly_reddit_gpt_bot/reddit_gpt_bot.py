from collections import defaultdict
import praw
import openai
import time
import csv
import random
import time
import datetime
import os
# Use the provided credentials
openai.api_key = os.environ.get("OPENAI_API_KEY")


def write_to_csv(engagement_data):
    with open("engagement_data.csv", "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["subreddit", "post_title",
                      "post_content", "post_id", "response", "timestamp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow(engagement_data)


client_id = ''  # Your Reddit client ID
client_secret = '' # Your Reddit client secret
user_agent = '' # Your Reddit user agent
username = ''  # Your Reddit username
password = ''  # Your Reddit password
subreddits = [
 "cscareerquestions", "datascience", 
]

# Initialize the Reddit API client
reddit_instance = praw.Reddit(client_id=client_id,
                              client_secret=client_secret,
                              user_agent=user_agent,
                              username=username,
                              password=password)
# Now you can use the Reddit API with the 'reddit' instance, e.g., get the top posts in a subreddit


def generate_system_variation():
    system_variations = [
        f"You are a witty Reddit poster and you want to engage with the community by posting comments on subreddits. You often reply in smart ways and often get upvoted, but don't use emojis or introduce yourself. You answer questions and give advice thoroughly. You should first engage directly with the original poster, address their comment, and provide a relevant response.",
        f"As a friendly Reddit user, you love sharing your experiences and knowledge with others. At the end of your response, you always mention that you are GPT-4.",
        # Add more system variations here
    ]

    return random.choice(system_variations)



def generate_response(prompt):
    system_variation = generate_system_variation()
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": system_variation},
            {"role": "user", "content": prompt}
        ]
    )
    generated_message = response.choices[0].message['content']
    message = generated_message.strip()
    return message


def engage_with_community(reddit_instance):
    engaged_posts = defaultdict(set)
    random.shuffle(subreddits)
    for subreddit_name in subreddits:
        try:
            subreddit = reddit_instance.subreddit(subreddit_name)
            submissions = list(subreddit.hot(limit=20))
        except Exception as e:
            continue
        random.shuffle(submissions)

        for submission in submissions:
            if not submission.stickied and submission.id not in engaged_posts[subreddit_name]:
                post_content = submission.selftext
                post_prompt = f"The post title is: {submission.title}. The post body is {post_content}. Please reply to this post:"

                try:
                    post_response = generate_response(post_prompt)
                    submission.reply(post_response)
                except Exception as e:
                    break

                engagement_data = {
                    "subreddit": subreddit_name,
                    "post_title": submission.title,
                    "post_id": submission.id,
                    "post_content": post_content,
                    "response": post_response,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                write_to_csv(engagement_data)
                print(f"Replied to post: {submission.title}")
                print(f"Response: {post_response}")

                engaged_posts[subreddit_name].add(submission.id)

                time.sleep(60 * 6) # don't spam too hard
                break


engage_with_community(reddit_instance)
