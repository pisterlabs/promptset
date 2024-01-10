import requests
import time
import random
import openai


# Authentication
openai.api_key = ''  # add your OpenAI API Key here
wordpress_jwt = ''  # add your wordpress JWT here
wp_endpoint = 'https://yourwordpresssite.com/wp-json/wp/v2/posts'  # add your wordpress endpoint here

# insert topics for your blogs here
blog_topics = [
    'business',
    'technology',
    'politics',
    # make sure it follows the same format as above
]

def make_blog(topic):
    print(f"Generating blog content on the topic: {topic}")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI that generates blog posts. Write a comprehensive blog about the given topic."},
            {"role": "user", "content": f"I want a blog about {topic}."},
        ],
        max_tokens=1000,
    )
    content = response.choices[0].message['content'].strip()
    print(f"Blog content generated: {content[:100]}...")  # Print the first 100 characters
    return content

def make_title(blog_content):
    print("Generating blog title...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI that generates catchy blog post titles. Generate a title that captures the essence of the given content. The title should be about five words maximum."},
            {"role": "user", "content": f"Here is the blog content:\n\n{blog_content}\n\n"},
        ],
        max_tokens=60,  # Usually, blog titles aren't very long, so 60 tokens should be enough.
    )
    title = response.choices[0].message['content'].strip()
    print(f"Blog title generated: {title}")
    return title

def post_to_wordpress(title, content):
    print("Posting to WordPress...")
    endpoint = wp_endpoint
    headers = {
        'Authorization': f'Bearer {wordpress_jwt}',
        'Content-Type': 'application/json'
    }

    data = {
        'title': title,
        'content': content,
        'status': 'draft',
    }
    response = requests.post(endpoint, headers=headers, json=data)

    if response.status_code != 201:
        print(f"Unable to publish blog as a draft")
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.content}")
    else:
        print(f"Published '{title}' as a draft")

    return response.status_code == 201


# Main driver function
def main():
    while True:
        print("Starting a new iteration...")
        topic = random.choice(blog_topics)
        blog_content = make_blog(topic)
        blog_title = make_title(blog_content)
        post_to_wordpress(blog_title, blog_content)
        time.sleep(3600)  # Pause for an hour (in seconds), you can change this to whatever you want, this allows you to control how many blogs are writen, you can remove this and have blogs being written forever, but I recommend against that. Remember you pay for each OpenAI call you receive.

if __name__ == "__main__":
    main()
