import openai
import os
import csv
from datetime import datetime
from openai import OpenAI
from pathlib import Path

# Ensure your OpenAI API key is set
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

def generate_facebook_posts(topic, text_content, tone):
    facebook_post_directive = f"""
    Please generate a list of unique, engaging, and attention-grabbing Facebook posts related to the topic of "{topic}". They should be appropriately hashtagged, with emojis, and reflect the "{tone}" voice consistent with our brand.
    And pull insights or action items from the {text_content}
    """

    facebook_post_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a social media manager."},
            {"role": "user", "content": facebook_post_directive},
        ],
    )

    facebook_posts = facebook_post_response.choices[0].message.content
    print("\nFacebook posts:")
    print(facebook_posts)
    
    return [post.strip() for post in facebook_posts.split('\n') if post.strip()]
def save_facebook_posts_to_csv(topic, facebook_posts):
    """
    Saves generated Facebook posts to a CSV file with version control.

    :param topic: The topic of the posts.
    :param facebook_posts: List of generated Facebook posts.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    directory = f"social_media_outputs/{today}"
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Handle the topic based on whether it's a string or a list
    if isinstance(topic, list):
        simplified_topic = topic[0]
    elif isinstance(topic, str):
        simplified_topic = topic.split(',')[0]
    else:
        raise TypeError("Topic must be a string or a list")

    version = 1
    file_path = f"{directory}/{simplified_topic}_facebook_posts_v{version}.txt"

    while Path(file_path).exists():
        version += 1
        file_path = f"{directory}/{simplified_topic}_facebook_posts_v{version}.txt"

    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Post'])
        for post in facebook_posts:  # Directly iterate over the list of posts
            writer.writerow([post])

    print(f"Facebook posts saved at {file_path}")

# Example usage in your main script
# facebook_posts = generate_facebook_posts(topic, blog_post, 'Inspirational')
# if facebook_posts:
#     save_facebook_posts_to_text(topic, facebook_posts)
