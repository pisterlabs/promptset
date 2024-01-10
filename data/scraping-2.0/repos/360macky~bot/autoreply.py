#!/usr/local/bin/python3

import tweepy
import logging
import openai
from utils import sent_notification_to_owner, get_tweepy_api, format_tweet
from decouple import config
import time

logger = logging.getLogger().setLevel(logging.INFO)

GPT_SYSTEM_INSTRUCTIONS_MENTION = "You are a fun bot that answers people's questions very briefly and irreverently. At the end you place an emoji. If they ask you in Spanish, answer in Spanish!"
GPT_MODEL='gpt-3.5-turbo'
INTERVAL = 60

openai.api_key = config("OPENAI_API_KEY")

def get_answer(username: str, question: str, previous_conversation: dict):
    """
    Get answer from GPT-3.5-turbo using OpenAI API
    """

    # If previous_conversation is not None, we need to add the previous conversation to the GPT-3.5-turbo
    # instructions
    if previous_conversation is not None:
        logging.info(f"Bot will reply with the context of a previous conversation")
        original_question = previous_conversation['original_question']
        bot_previous_answer = previous_conversation['bot_previous_answer']
        messages = [
            {"role": "system", "content": GPT_SYSTEM_INSTRUCTIONS_MENTION},
            {"role": "user", "content": f"Hi, my name is {username}, I'm asking: {original_question}?"},
            {"role": "assistant", "content": f"{bot_previous_answer}"},
            {"role": "user", "content": f"{question}?"}
        ]
    else:
        logging.info(f"Bot will reply without context")
        messages = [
            {"role": "system", "content": GPT_SYSTEM_INSTRUCTIONS_MENTION},
            {"role": "user", "content": f"Hi, my name is {username}, I'm asking: {question}?"}
        ]


    try:
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=messages,
            max_tokens=70
        )
        logging.info(f"GPT successfully generated anser: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except:
        logging.error("Failed to get GPT response")
        return None

def remove_bot_mention(text):
    """
    Remove bot mention from text
    """
    return ' '.join(word for word in text.split() if not word.startswith('@360mackyBOT'))

def check_mentions(api, since_id):
    """
    Check mentions, if they exists, reply to them
    """
    logging.info("Checking new mentions")
    new_since_id = since_id

    # Get mentions
    for tweet in tweepy.Cursor(api.mentions_timeline, since_id=since_id).items():
        
        user_question = remove_bot_mention(tweet.text)

        # Check if user question is empty
        if user_question == "" or user_question == " ":
            logging.info(f"User question is empty. Continue with the process...")
            continue

        # Check if we have already liked this tweet
        if tweet.favorited:
            continue
        else:
            logging.info(f"Liking {tweet.text}. Continue with the process...")
            tweet.favorite()

        # Update the since_id
        new_since_id = max(tweet.id, new_since_id)

        # If the tweet is a reply, maybe it's a continuation of a previous conversation
        # or maybe it's a question to the bot itself. So let's check that.
        previous_conversation = None

        if tweet.in_reply_to_status_id is not None:
            # Get the previous tweet:
            previous_tweet = api.get_status(tweet.in_reply_to_status_id)
            # Is the previous tweet from the bot?
            if previous_tweet.user.screen_name == "360mackyBOT":
                logging.info(f"Bot is replying to a previous conversation")
                bot_previous_answer = remove_bot_mention(previous_tweet.text)
                
                # Get the original question:
                original_question = api.get_status(previous_tweet.in_reply_to_status_id)

                # Wrap the original question and the previous answer in a dict "previous_conversation":
                previous_conversation = {
                    "original_question": original_question.text,
                    "bot_previous_answer": bot_previous_answer
                }


        if not tweet.user.following:
            tweet.user.follow()


        logging.info(f"Answering question of {user_question} to {tweet.user.name}")

        generated_answer = format_tweet(get_answer(tweet.user.name, user_question, previous_conversation))

        if generated_answer is None:
            sent_notification_to_owner(f"🤖 *Marcelo Bot* failed to generated an answer to *{user_question}*")
            continue
        
        try:
            api.update_status(
            status=generated_answer,
            in_reply_to_status_id=tweet.id,
            auto_populate_reply_metadata=True,
            )
            sent_notification_to_owner(f"🤖 *Marcelo Bot* answered the question *{user_question}* to *{tweet.user.name}* with *{generated_answer}*")
        except:
            sent_notification_to_owner(f"🤖 *Marcelo Bot* failed to answer the question *{user_question}* to *{tweet.user.name}*")

    return new_since_id

def main():
    api = get_tweepy_api()
    since_id = 1
    while True:
        since_id = check_mentions(api, since_id)
        logging.info("Waiting for new mentions...")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
