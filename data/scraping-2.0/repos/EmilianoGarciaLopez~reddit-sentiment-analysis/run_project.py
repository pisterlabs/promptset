import os
import openai

import joblib
import praw
from decouple import config

# Fetch comments of a user from the Reddit API
from utils.models import predict_from_clf

reddit = praw.Reddit(
    user_agent=config("USER_AGENT"),
    client_id=config("CLIENT_ID"),
    client_secret=config("CLIENT_SECRET"),
    username=config("USERNAME"),
    password=config("PASSWORD"),
)

openai.api_key = config("GPT3_API_KEY")


# Fetch comments of a user from the Reddit API
def get_user_comments(user: str, num_comments: int = 50) -> list:
    usr_comments = []
    for comment in reddit.redditor(user).comments.new(limit=num_comments):
        usr_comments.append(comment.body)
    return usr_comments


def get_predictions(comments: list, model, model_folder='models') -> list:
    comments = [c.lower() for c in comments]
    clf, count_vect, tfidf_transformer = joblib.load(
        os.path.join(model_folder, model))

    return predict_from_clf(comments, clf, count_vect, tfidf_transformer)


def classify_user(comments_preds, thold=0.5) -> tuple:
    ones = sum(comments_preds)
    p = ones / len(comments_preds)

    if p > thold:
        return "democrat", p
    else:
        return "conservative", 1 - p


def main():
    user = input("Enter a username: ")
    runtime_comments = get_user_comments(user)

    preds = get_predictions(runtime_comments, 'NV_v1.z')
    classification, prob = classify_user(preds)

    print(f'Class: {classification}, p: {prob}')


def useGPT3():
    user = input("Enter a username: ")
    runtime_comments = get_user_comments(user)
    conservative_count = 0
    democrat_count = 0
    for comment in runtime_comments:
        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt=f"Is the following text written by a democrat or a conservative?\n {comment}",
            temperature=0,
            max_tokens=10,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        # prints openai response
        if "conservative" in response.choices[0].text:
            conservative_count += 1
        if "democrat" in response.choices[0].text:
            democrat_count += 1

    if conservative_count > democrat_count:
        print("Class: conservative, p: ", conservative_count / len(runtime_comments))
    else:
        print("Class: democrat, p: ", democrat_count / len(runtime_comments))


if __name__ == "__main__":
    main()
    # useGPT3()