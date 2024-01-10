import requests
import urllib3
import time
import streamlit as st
import nltk
import openai

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

POSTS_PER_BATCH = 30  # Number of posts to publish at a time


def generate_blog_articles(keywords, topics, tone, openai_api_key):
    blog_articles = []

    for keyword in keywords:
        for topic in topics:
            title = generate_title(keyword, topic, openai_api_key)

            prompt = f"Write a {tone} blog article with the title: {title}\n\n"

            # Generate the article using Langchain or GPT model
            openai.api_key = openai_api_key
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=1000,
                n=1,
                stop=None,
                temperature=0.7,
            )
            article = response.choices[0].text.strip()

            # Append the generated article to the list
            blog_articles.append(
                {
                    "title": title,
                    "article": article,
                }
            )

            # Introduce a delay between each API call
            time.sleep(5)

    return blog_articles


def publish_articles_on_wordpress(
    blog_articles, category_name, wordpress_domain, admin_username, admin_password
):
    # Authenticate with WordPress using admin username and password
    login_url = f"https://{wordpress_domain}/wp-json/jwt-auth/v1/token"
    login_data = {"username": admin_username, "password": admin_password}
    login_response = requests.post(login_url, params=login_data, verify=False)
    if login_response.status_code != 200:
        st.error(
            "Failed to authenticate with WordPress. Please check your admin username and password."
        )
        return
    else:
        st.success("Login was successful, about to post the articles!")

    token = login_response.json().get("token")

    # Get the category ID based on the selected category name
    category_id = get_category_id(category_name, wordpress_domain)

    if category_id is None:
        st.error(f"Failed to find category ID for category: {category_name}")
        return

    # Publishing URL
    url = f"https://{wordpress_domain}/wp-json/wp/v2/posts"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Split the blog_articles into chunks and publish
    for i in range(0, len(blog_articles), POSTS_PER_BATCH):
        publish_batch(blog_articles[i : i + POSTS_PER_BATCH], url, headers, category_id)


def publish_batch(blog_articles, url, headers, category_id):
    # Publish blog articles on WordPress website
    for i, article_info in enumerate(blog_articles, start=1):
        title = article_info["title"]
        article = article_info["article"]

        data = {
            "title": title,
            "content": article,
            "status": "publish",
            "categories": [category_id],  # Pass category ID as a list
        }

        response = requests.post(url, json=data, headers=headers, verify=False)
        if response.status_code == 201:
            st.success(f"Article {i} published successfully!")
        else:
            st.error(f"Failed to publish Article {i}. Error: {response.text}")

        time.sleep(2)  # Wait between each article to avoid overloading the server


def generate_title(keyword, topic, openai_api_key):
    prompt = f"Generate a title for a blog article about {keyword.strip()} and {topic.strip()}. Keep the title within 50 tokens."

    # Generate the title using the GPT model
    openai.api_key = openai_api_key
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    title = response.choices[0].text.strip().replace('"', "")

    return title


def get_categories(wordpress_domain):
    categories_url = f"https://{wordpress_domain}/wp-json/wp/v2/categories?per_page=100"
    response = requests.get(categories_url)
    categories = response.json()
    return categories


def get_category_id(category_name, wordpress_domain):
    categories_url = f"https://{wordpress_domain}/wp-json/wp/v2/categories?per_page=100"
    response = requests.get(categories_url)
    categories = response.json()

    for category in categories:
        if category["name"] == category_name:
            return category["id"]

    return None


def main():
    # Set page title
    st.title("Blog Post Generator")

    # Download nltk.punkt tokenizer if not already downloaded
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # User input section
    st.subheader("Enter Your Input")
    keywords = st.text_input("Keywords (separated by comma)")
    topics = st.text_input("Topics (separated by comma)")
    tone = st.selectbox("Tone", ["Funny", "Serious", "Informative"])

    # Create fields for the OpenAI API key, WordPress domain, and login credentials
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    wordpress_domain = st.text_input("WordPress Domain")
    admin_username = st.text_input("WordPress Admin Username")
    admin_password = st.text_input("WordPress Admin Password", type="password")

    # Retrieve categories
    categories = get_categories(wordpress_domain)
    category_names = [category["name"] for category in categories]

    # Create dropdown for selecting category
    category_name = st.selectbox("Category", category_names)

    if st.button("Generate and Publish"):
        # Convert keywords and topics to lists
        keywords = [keyword.strip() for keyword in keywords.split(",")]
        topics = [topic.strip() for topic in topics.split(",")]

        # Generate blog articles
        blog_articles = generate_blog_articles(keywords, topics, tone, openai_api_key)

        # Publish blog articles on WordPress in batches
        publish_articles_on_wordpress(
            blog_articles,
            category_name,
            wordpress_domain,
            admin_username,
            admin_password,
        )


if __name__ == "__main__":
    main()
