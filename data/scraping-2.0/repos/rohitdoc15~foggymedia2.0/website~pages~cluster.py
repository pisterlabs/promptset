import sys
import time
import os
import django
from django.utils import timezone
from datetime import timedelta
from fuzzywuzzy import fuzz
import openai
from collections import Counter as CollectionsCounter
sys.path.append('/home/rohit/news/website')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'website.settings')
django.setup()

from pages.models import NewsChannel, Video, TrendingTopic



stopwords = ['gujarat', 'abp news' , 'India News' ,'Top Headlines ' , 'WION']



now = timezone.now()
time_4_hours_ago = now - timedelta(hours=24)

openai.api_key = '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'

channels = NewsChannel.objects.all()
all_topics = [video.topic for video in Video.objects.all()]

# Calculate the frequency of each topic
topic_counter = CollectionsCounter(all_topics)

# Filter out blank and dash ("-") topics
for topic in list(topic_counter):  # Use list to avoid 'dictionary changed size during iteration' error
    if topic == "" or topic == "-":
        del topic_counter[topic]

# Get the 5 most common topics
most_common_topics = topic_counter.most_common(5)
print(f"Most common topics: {most_common_topics}")

topics_str = ' '.join([f'{i+1}. {topic[0]}' for i, topic in enumerate(most_common_topics)])
for channel in channels:
    print(f"Processing channel: {channel.name}")

    videos = Video.objects.filter(channel=channel, published_date__range=(time_4_hours_ago, now))
    titles = [video.title for video in videos]
    print(f"Found {len(titles)} videos for channel")

    all_titles = '\n'.join(titles)
    max_len = 5000
    if len(all_titles) > max_len:
        all_titles = all_titles[:max_len]
    
    top_trending_topics = TrendingTopic.objects.order_by('rank')[:5]
   
    system_message = f"You are a sophisticated AI model trained in topic extraction and text analysis. Please give the topics in the format example: {topics_str}"

    for attempt in range(3):  # Increase to 3 to allow 2 retries
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"I have a list of news titles from the last 4 hours: {all_titles}. Can you analyze them and tell me the five main topics that these titles seem to be about? The topics should be in Title Case without hashtags,and should be ordered by frequency. "},
                ],
                temperature=0,
              
            )
            reply = completion.choices[0].message['content']
            print(f"AI's reply: {reply}")
            if len(reply) > 200:
                print(f"Reply is too long ({len(reply)} characters). Retrying.")
                if attempt == 2:  # If this was the last attempt
                    print("After two attempts, reply is still too long. Updating only the trending topic in the file.")
                    # Code to update trending topic in the file goes here
                continue  # Retry if reply was too long
            else:
                print("Successfully retrieved AI's topic analysis.")
                break  # Exit loop if reply was of acceptable length
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(1)

    
    channel_topics = reply.split('\n')
    channel_topics = [topic.split('. ')[1] if '. ' in topic else topic for topic in channel_topics]


    # Fetch the old topics from the last two days
    time_2_days_ago = now - timedelta(days=2)

    old_topics = list(Video.objects.filter(published_date__range=(time_2_days_ago, now)).values_list('topic', flat=True))
    topic_counts = CollectionsCounter(old_topics)
    similarity_threshold = 70  # You can adjust this value according to your needs
    old_topics = sorted(topic_counts, key=topic_counts.get, reverse=True)

    # List of certain words
    certain_words = ['Controversy', 'Updates', 'news' ,'Politics']

    for i, topic in enumerate(channel_topics):
        # Check if the topic contains any certain words
        has_certain_word = any(word in topic for word in certain_words)

        if has_certain_word:
            # Adjust similarity threshold for topics with certain words
            similarity_threshold = 80  # Adjust the value as needed for topics with certain words

        # Check if the topic already exists in the old topics using fuzzy matching
        similar_old_topic = next((old_topic for old_topic in old_topics if fuzz.ratio(topic, old_topic) > similarity_threshold), None)

        if similar_old_topic:
            # If similar old topic found, replace new topic with old topic
            print(f"New topic '{topic}' is similar to old topic '{similar_old_topic}', replacing.")
            topic = similar_old_topic

        # Update the channel topics with the new or old topic
        channel_topics[i] = topic

    trending_topics = [t.topic for t in TrendingTopic.objects.all()]

    all_topics = set(trending_topics + channel_topics)

    print(f"Total topics (trending + channel specific): {len(all_topics)}")

    # Store unique topics
    unique_topics = []

    for video in videos:
        video_topic = None
        max_similarity = -1
        for topic in all_topics:
            similarity = fuzz.ratio(video.title, topic)
            if similarity > similarity_threshold and similarity > max_similarity:
                video_topic = topic
                max_similarity = similarity
        if video_topic:
            video.topic = video_topic
            video.save()
            print(f"Assigned topic '{video_topic}' to video '{video.title}'")

            # Compare the new topic with each trending topic
            is_unique = True
            for trending_topic in trending_topics:
                if fuzz.ratio(video_topic, trending_topic) > similarity_threshold:
                    is_unique = False
                    break

            if is_unique:
                unique_topics.append(video_topic)

    # Save unique topics to a file
            
    trending_topics_set = set(trending_topics)
    channel_topics_set = set(channel_topics)



    # Get the unique topics from the channel topics when compared to trending topics

    unique_channel_topics = []
    for channel_topic in channel_topics_set:
        # Check if any keyword from the channel topic is present in the trending topics
        has_common_keyword = any(any(keyword.lower() in trending_topic.lower() for keyword in channel_topic.split()) for trending_topic in trending_topics)
        if not has_common_keyword:
            unique_channel_topics.append(channel_topic)



    # Merge unique channel topics and all trending topics
    unique_topics = unique_channel_topics + list(trending_topics_set)
    filtered_topics = [topic for topic in unique_topics if topic.lower() not in stopwords]

    # Save unique topics to a file
    with open(f"{channel.name}.txt", 'w') as file:
        for topic in filtered_topics:
            file.write(f"{topic}\n")
    print(f"Saved filtered topics to file: {channel.name}.txt")