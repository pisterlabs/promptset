import sys
import time
sys.path.append('/home/rohit/news/website')
from googletrans import Translator
translator = Translator()
import os
import django
from django.utils import timezone
from datetime import timedelta

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'website.settings')  
django.setup()
from collections import Counter as CollectionsCounter
from pages.models import NewsChannel, Video, TrendingTopic
import openai
from fuzzywuzzy import fuzz

# Get the current time
now = timezone.now()

# Get the time 4 hours ago and 2 days ago
time_4_hours_ago = now - timedelta(hours=8)
time_2_days_ago = now - timedelta(days=2)

openai.api_key = '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'

# Fetch all video titles from your database that were published in the last 4 hours
videos = Video.objects.filter(published_date__range=(time_4_hours_ago, now))
channels = NewsChannel.objects.all()

titles = []
for channel in channels:
    # Fetch the latest 5 video titles from this channel that were published in the last 4 hours
    videos = Video.objects.filter(channel=channel, published_date__range=(time_4_hours_ago, now)).order_by('-published_date')[:5]
    # Append these titles to our master list
    for video in videos:
        title = video.title
        # Translate the title to English
        translated_title = translator.translate(title, dest='en').text
        titles.append(translated_title)

# Fetch the topics from all videos
all_topics = [video.topic for video in Video.objects.all()]

# Calculate the frequency of each topic
topic_counter = CollectionsCounter(all_topics)

# Filter out blank and dash ("-") topics
for topic in list(topic_counter):  # Use list to avoid 'dictionary changed size during iteration' error
    if topic == "" or topic == "-":
        del topic_counter[topic]

# Get the 5 most common topics from TrendingTopic model
most_common_topics = TrendingTopic.objects.order_by('rank')[:5]

# Convert most_common_topics to a list of tuples for consistency with your old code
most_common_topics = [(topic.topic, topic.rank) for topic in most_common_topics]

print(f"Most common topics: {most_common_topics}")
topics_str = ' '.join([f'{i+1}. {topic[0]}' for i, topic in enumerate(most_common_topics)])
system_message = f"You are a sophisticated AI model trained in news topic extraction. Please give the topics in the format example: {topics_str}"

all_titles = '\n'.join(titles)

# Remember to limit the length of the input as the API has a maximum token limit
max_len = 10000
if len(all_titles) > max_len:
    all_titles = all_titles[:max_len]
print(all_titles)

# Retry up to 5 times
for retry in range(5):
    while True:  # Continue until we get a reply under 160 characters
        try:
            # Construct the conversation with the AI model
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"I have a list of news titles from the last 4 hours: {all_titles}. Can you analyze them and tell me the five unique topics that these titles seem to be about? The topics should be in Title Case without hashtags, and should be ordered by frequency. "},
                ],
                
            )
            # Extract the model's reply
            reply = completion.choices[0].message['content']

            # Check if the reply is under 160 characters
            if len(reply) <= 200:
                break
            print(f"Generated reply was too long, retrying...")
            print(f"Generated reply: {reply}")
            time.sleep(5)  # Wait for 5 seconds before retrying
        except Exception as e:
            print(f"Error on attempt {retry+1}: {e}")
            

    # If we got a reply under 160 characters, break out of the retry loop
    if len(reply) <= 200:
        break
    print(f"Retry #{retry+1} failed, waiting 5 seconds before next attempt...")
    time.sleep(5)  # Wait for 5 seconds before next retry

# Split the reply into topics
topics = reply.split('\n')
# Remove numbers and dots from the topics
topics = [topic.split('. ')[1] if '. ' in topic else topic for topic in topics]

# Fetch the old topics from the last two days
old_topics = Video.objects.filter(published_date__range=(time_2_days_ago, now)).values_list('topic', flat=True)

# Set a threshold for the similarity
similarity_threshold = 50

# Update the TrendingTopic model
# Update the TrendingTopic model
for i, (topic, _) in enumerate(most_common_topics):
    # Initialize max_similarity and similar_old_topic
    max_similarity = -1
    similar_old_topic = None

    # Check if the topic already exists in the old topics using fuzzy matching
    for old_topic in old_topics:
        similarity = fuzz.ratio(topic, old_topic)
        if similarity > max_similarity:
            max_similarity = similarity
            similar_old_topic = old_topic

    if max_similarity > similarity_threshold:
        # If similar old topic found with the highest similarity score, replace new topic with old topic
        print(f"New topic '{topic}' is similar to old topic '{similar_old_topic}' with a similarity score of {max_similarity}, replacing.")
        topic = similar_old_topic

    # Update the TrendingTopic model with the new or old topic
    trending_topic, created = TrendingTopic.objects.get_or_create(rank=i+1)
    trending_topic.topic = topic
    trending_topic.save()


