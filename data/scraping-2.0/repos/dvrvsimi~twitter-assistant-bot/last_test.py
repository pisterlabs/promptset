import openai
import tweepy
import time
from datetime import datetime, timedelta

from config import *

# Set up OpenAI API credentials
openai.api_key = openai_key

# authorizing with tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


model_engine = "text-davinci-003"

# function to generate a fact about the given topic
def generate_fact(topic):
    prompt = f"you are a grumpy computer programmer, tell a fact about {topic} in a rude and sarcarstic tone"
    response = openai.Completion.create(
        # test gpt-3.5-turbo-0301 for engine
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.8,
    )
    fact = response.choices[0].text.strip()
    return fact

# define a function to translate text from one language to another
def translate_text(text, target_language):
    translation = openai.Completion.create(
        engine=model_engine,
        prompt=f"translate the following text into {target_language}: '{text}'",
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.3,
    )
    return translation.choices[0].text.strip()

# Define a function to send a reminder tweet
def send_reminder(task, days_until_reminder):
    # Calculate the date of the reminder
    reminder_date = datetime.now() + timedelta(days=days_until_reminder)
    formatted_date = reminder_date.strftime("%B %d, %Y")
    # Create the reminder message
    reminder_message = f"remember {task} on {formatted_date}?"
    # Send the tweet
    api.update_status(reminder_message)

# Define a function to check for new mentions and handle them
def handle_mentions():
    # Get the most recent mention
    mentions = api.mentions_timeline(count=1)
    if len(mentions) == 0:
        return
    mention = mentions[0]
    # Parse the mention text and extract the command and arguments
    mention_text = mention.text.casefold()
    for mention in mentions:
        if mention.in_reply_to_status_id is not None:  # Skip replies to other tweets
            continue
        if mention.user.screen_name == "dvrvsimi":  # Skip mentions from self
            continue
        
        # Parse mention text and extract topic
        if "tell a fact about" not in mention_text:
            continue
        topic = mention_text.split("tell a fact about")[1].strip()
        
        # generate fact
        try:
            fact = generate_fact(topic)
            # Post the fact as a reply to the mention
            api.update_status(f"@{mention.user.screen_name} {fact}", in_reply_to_status_id=mention.id)
        except openai.OpenAIError as e:
            # Handle OpenAI API errors
            api.update_status(f"@{mention.user.screen_name} sorry, an error occurred while processing your request.", in_reply_to_status_id=mention.id)
        except tweepy.TweepyException as e:
            # Handle Twitter API errors
            api.update_status(f"@{mention.user.screen_name} sorry, an error occurred while processing your request.", in_reply_to_status_id=mention.id)
        except Exception as e:
            # Handle other errors
            api.update_status(f"@{mention.user.screen_name} sorry, an error occurred while processing your request.", in_reply_to_status_id=mention.id)

        #for translation fUnction
        if "translate" in mention_text:
            command, text, target_language = mention_text.split()
            if command == "@dvrvsimi" and text and target_language:
                try:
                    translation = translate_text(text, target_language)
                    reply_message = f"translation: {translation}"
                except Exception as e:
                    print(f"error translating text: {e}")
                    reply_message = "an error occurred while translating the text."
                api.update_status(
                    status=reply_message,
                    in_reply_to_status_id=mention.id,
                    auto_populate_reply_metadata=True
                )
        # for reminder function
        elif "reminder" in mention_text:
            command, task, days_until_reminder = mention_text.split()
            if command == "@dvrvsimi" and task and days_until_reminder:
                try:
                    days_until_reminder = int(days_until_reminder)
                    send_reminder(task, days_until_reminder)
                    reply_message = f"reminder set! You will be reminded in {days_until_reminder} days to complete {task}."
                except ValueError:
                    reply_message = "invalid reminder format, please use this format: @ dvrvsimi reminder {task} in {days_until_reminder} days \n here's an example: @ dvrvsimi reminder push code in 2 days"
                except Exception as e:
                    print(f"error setting reminder: {e}")
                    reply_message = "an error occurred while setting the reminder."
                api.update_status(
                    status=reply_message,
                    in_reply_to_status_id=mention.id,
                    auto_populate_reply_metadata=True
                )


if __name__ == "__main__":
    while True:
        try:    
            handle_mentions()
        except tweepy.TweepyException as e:
            # Handle Twitter API errors
            print("an error occurred while handling mentions:", e)
        except Exception as e:
            # Handle other errors
            print("an error occurred while handling mentions:", e)
        # Wait for 2 minutes seconds before checking for new mentions
        time.sleep(120)