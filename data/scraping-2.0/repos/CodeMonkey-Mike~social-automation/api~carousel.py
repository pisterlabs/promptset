
import openai
import json
import random
from datetime import datetime, timezone
from flask import jsonify
from flask_sqlalchemy import SQLAlchemy
from .models import Carousel, CarouselSlides, CarouselTopics
from .twitter_poll import client

db = SQLAlchemy()

defaultPrompts = {
    "role": "user",
    "content": """I talk on social media about the following topics - stocks, stock market, entrepreneurship, building wealth, motivation, and productivity, and self improvement.\n\nCan you choose a topic and summarize it into a Twitter thread? But your response in JSON format"""
}


def get_chat_message(article_content):
    prompts = [
        {
            "role": "system",
            "content": """I will provide you with some text from an article, and you will make a Twitter thread about it.  Here are your instructions on how to accomplish that:\n\n> Please make between 5 - 7 tweets for this thread, but do not add hashtags to the content.\n> The first tweet should hook the reader with curiosity.  When the reader reads it, they should think things like, "then what happened" or "I have to know more!" or "what is going on?" An open loop should be created so that the reader will want to read the next tweet\n> The first sentence of the first tweet should be in UPPER case letters. but the second should be normal case, and also be short and brief.  \n> One main keyword in the first sentence should be encapsulated in *asterisk* characters.\n> Separate them by Tweet 1, Tweet 2, etc.. \n> From the second tweet onward, make sure there are line breaks after sentences to make them easier to read.  Make the sentences short.  Include bullet points where possible. Be sure to include emojis\n> Make sure each tweet is less than 270 characters.\n> Make the response in JSON format where each tweet has a property called "HashTag" and put the hashtags into an array in this property instead of in the content. DO NOT add hashtags to the content property in the JSON.\n\nHere is the required JSON format for the response:\n{\n  "Thread": [\n    {\n      "TweetNumber": 1,\n      "Content": "value",\n      "HashTag": ["value", "value"],\n      }\n    }\n  ]\n}\n> Do not deviate from this format\n\nEvery time I paste in a new article, you will generate response in JSON format"""
        },
    ]
    if article_content and len(article_content):
        prompts.append({
            "role": "user",
            "content": f"""New article: {article_content}"""
        })
    else:
        prompts.append(defaultPrompts)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompts,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # message = response["choices"][0]["message"]
    # prompt.append(message)
    content = response["choices"][0]["message"]["content"]
    content_dict = json.loads(content)
    return content_dict


def get_carousel_topics():
    try:
        # Fetch all data from the 'topics' table
        topics = CarouselTopics.query.all()
        # Convert the data to a list of dictionaries
        data_list = [{"id": topic.id, "content": topic.content}
                     for topic in topics]

        return random.choice(data_list)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_carousel():
    try:
        # Fetch all data from the 'topics' table
        carousels = Carousel.query.all()
        # Convert the data to a list of dictionaries
        data_list = [{"id": carousel.id, "created_at": carousel.created_at}
                     for carousel in carousels]
        return data_list

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_carousel_slides(carousel_id):
    try:
        # Fetch all data from the 'topics' table
        carouselSlides = CarouselSlides.query.filter_by(
            carousel_id=carousel_id).all()
        # Convert the data to a list of dictionaries
        data_list = [{"id": carouselSlide.id, "description": carouselSlide.description, "hashtag": carouselSlide.hashtag, "sequence": carouselSlide.sequence}
                     for carouselSlide in carouselSlides]

        return data_list

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def delete_carousel_topic(topic):
    topic_id = topic.get("id")
    topic_to_delete = db.session.query(CarouselTopics).get(topic_id)
    db.session.delete(topic_to_delete)
    db.session.commit()
    return 'Deleted topic.'


def save_carousel(title):
    try:
        # Create a new Carousel instance
        new_carousel = Carousel(title=title)

        # Add the new carousel to the database
        db.session.add(new_carousel)
        db.session.commit()

        # Retrieve the committed carousel from the database
        committed_carousel = Carousel.query.get(new_carousel.id)

        return {
            "message": "Poll created successfully!",
            "carousel_id":  committed_carousel.id
        }
    except Exception as e:
        # Handle exceptions appropriately (e.g., log the error)
        return jsonify({"error": str(e)}), 500


def save_carousel_slides(description, hashtag, sequence, carousel_id):
    try:
        print(description, hashtag, sequence, carousel_id)
        # Create a new Carousel instance
        new_slide = CarouselSlides(
            description=description, hashtag=hashtag, sequence=sequence, carousel_id=carousel_id)

        # Add the new carousel to the database
        db.session.add(new_slide)
        db.session.commit()

        return jsonify({
            "message": "Poll created successfully!",
        }), 201
    except Exception as e:
        # Handle exceptions appropriately (e.g., log the error)
        return jsonify({"error": str(e)}), 500


def get_tweet_and_save_to_db():
    topic = get_carousel_topics()
    article = topic.get('content')
    thread = get_chat_message(article)
    tweets = thread['Thread']
    saved_carousel = save_carousel(tweets[0]['Content'])
    saved_carousel_id = saved_carousel.get('carousel_id')
    if len(tweets) > 0:
        for tweet in tweets:
            hashtags = ",".join(tweet['HashTag'])
            save_carousel_slides(
                tweet['Content'], hashtags, tweet['TweetNumber'], saved_carousel_id)

    delete_carousel_topic(topic)


def post_to_twitter_in_thread(tw_text):
    if len(tw_text) > 0:
        response = client.create_tweet(
            text=f"{tw_text}",
        )
        # Access attributes directly
        tweet_data = response.data
        return tweet_data
    else:
        print('Error when posting new tweet.')
        return None


def reply_to_tweet_by_id(tw_text, tweet_id):
    if len(tw_text) > 0:
        client.create_tweet(
            text=f"{tw_text}",
            in_reply_to_tweet_id=tweet_id
        )
    else:
        print('Error when reply to tweet.')
        return None

# Random pick a carousel < 7 days


def pick_carousel_more_than_7_days(slides):
    # Get the current date and time
    current_date = datetime.utcnow().replace(tzinfo=timezone.utc)
    # Filter items less than 7 days old
    filtered_items = [
        item for item in slides if (
            current_date - item['created_at']
        ).days > 7
    ]
    return random.choice(filtered_items)

#  Delete a carousel


def delete_posted_carousel(carousel_id):
    print('carousel_id:', carousel_id)
    carousel_to_delete = db.session.query(Carousel).get(carousel_id)
    db.session.delete(carousel_to_delete)
    db.session.commit()
    return 'Deleted carousel.'
