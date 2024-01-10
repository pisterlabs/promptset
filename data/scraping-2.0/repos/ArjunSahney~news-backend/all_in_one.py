# Returns dictionary (headlines_by_category) with category, headline, and summaries

import openai
import requests
from newsapi import NewsApiClient

# Initialize API keys
newsapi_key = '785379e735a84282af1c6b35cf335a59'
openai_api_key = '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'

# Set up NewsApiClient and OpenAI
newsapi = NewsApiClient(api_key=newsapi_key)
openai.api_key = openai_api_key

def summarize_headline_5_words(article):
    """Summarizes the article headline into 5 words."""
    message = {
        "role": "system",
        "content": "You are a helpful assistant. Summarize headlines into 5 words."
    }
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            message,
            {"role": "user", "content": f"Please summarize this headline: {article}"}
        ]
    )
    summarized_text = response['choices'][0]['message']['content'].strip()
    return summarized_text

def fetch_articles_from_source(source, num_articles=100):
    """Fetch articles from a specified source."""
    articles_list = []
    pages_needed = (num_articles + 99) // 100
    for page in range(1, pages_needed + 1):
        response = newsapi.get_everything(sources=source, page_size=100, page=page, language='en')
        articles_list.extend(response.get('articles', []))
    return articles_list[:num_articles]

def fetch_and_store_articles_by_category_and_source():
    """Fetch and store articles by category and source."""
    sources_by_category = {
        "news": ["reuters", "associated-press", "axios"],
        "business": ["business-insider", "financial-times"],
        "technology": ["techcrunch", "the-verge", "engadget"]
    }
    num_articles_by_category = {
        "news": 100,
        "business": 100,
        "technology": 100
    }
    articles_by_category = {}
    for category, sources in sources_by_category.items():
        articles_by_category[category] = []
        num_articles_per_source = num_articles_by_category[category] // len(sources)
        for source in sources:
            articles_by_category[category].extend(fetch_articles_from_source(source, num_articles_per_source))
    return articles_by_category

# Fetch articles
articles_by_category = fetch_and_store_articles_by_category_and_source()

# Initialize a dictionary to store headlines organized by category
headlines_by_category = {"news": [], "business": [], "technology": []}

# Rest of your code remains the same up to the point where you start processing each article

# Process each article and print summaries as they are added
for category, articles in articles_by_category.items():
    print(f"Category: {category}")
    for article in articles:
        headline = article['title']
        summary = summarize_headline_5_words(headline)
        
        # Search for related articles
        url = "https://newsapi.org/v2/everything"
        params = {"q": summary, "apiKey": newsapi_key}
        response = requests.get(url, params=params)
        related_articles = response.json().get('articles', [])

        # Create a 75-word summary
        if related_articles:
            descriptions = ' '.join([a['description'] for a in related_articles if a['description']])
            topic_prompt = f"Summarize the following into a 75-word summary:\n\n{descriptions}"
            perspective_prompt = "Identify and summarize the two major perspectives with specifics in 50 words each based on the following content:\n\n" + descriptions

            # Summarize topic
            topic_summary = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": topic_prompt}]
            )['choices'][0]['message']['content'].strip()

            # Summarize perspectives
            perspective_summary = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": perspective_prompt}]
            )['choices'][0]['message']['content'].strip()

            # Print the information immediately
            print(f"Headline: {headline}")
            print(f"Summary: {summary}")
            print(f"Topic Summary: {topic_summary}")
            print(f"Perspective Summary: {perspective_summary}\n")

        # Add the information to the dictionary
        headlines_by_category[category].append({
            'headline': headline,
            'summary': summary,
            'topic_summary': topic_summary,
            'perspective_summary': perspective_summary
        })


# Print the organized headlines by category
for category, headlines in headlines_by_category.items():
    print(f"Category: {category}")
    for headline_info in headlines:
        print(f"Headline: {headline_info['headline']}")
        print(f"Summary: {headline_info['summary']}")
        print(f"Topic Summary: {headline_info['topic_summary']}")
        print(f"Perspective Summary:\n{headline_info['perspective_summary']}\n")



#WORKING VERSION W/O dictionary 

# import openai
# import requests
# from newsapi import NewsApiClient

# # Initialize API keys
# newsapi_key = '785379e735a84282af1c6b35cf335a59'
# openai_api_key = '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'

# # Set up NewsApiClient and OpenAI
# newsapi = NewsApiClient(api_key=newsapi_key)
# openai.api_key = openai_api_key

# def summarize_headline_5_words(article):
#     """Summarizes the article headline into 5 words."""
#     message = {
#         "role": "system",
#         "content": "You are a helpful assistant. Summarize headlines into 5 words."
#     }
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             message,
#             {"role": "user", "content": f"Please summarize this headline: {article}"}
#         ]
#     )
#     summarized_text = response['choices'][0]['message']['content'].strip()
#     return summarized_text

# def fetch_articles_from_source(source, num_articles=100):
#     """Fetch articles from a specified source."""
#     articles_list = []
#     pages_needed = (num_articles + 99) // 100
#     for page in range(1, pages_needed + 1):
#         response = newsapi.get_everything(sources=source, page_size=100, page=page, language='en')
#         articles_list.extend(response.get('articles', []))
#     return articles_list[:num_articles]

# def fetch_and_store_articles_by_category_and_source():
#     """Fetch and store articles by category and source."""
#     sources_by_category = {
#         "news": ["reuters", "associated-press", "axios"],
#         "business": ["business-insider", "financial-times"],
#         "technology": ["techcrunch", "the-verge", "engadget"]
#     }
#     num_articles_by_category = {
#         "news": 100,
#         "business": 100,
#         "technology": 100
#     }
#     articles_by_category = {}
#     for category, sources in sources_by_category.items():
#         articles_by_category[category] = []
#         num_articles_per_source = num_articles_by_category[category] // len(sources)
#         for source in sources:
#             articles_by_category[category].extend(fetch_articles_from_source(source, num_articles_per_source))
#     return articles_by_category

# # Fetch articles
# articles_by_category = fetch_and_store_articles_by_category_and_source()

# # Process each article
# for category, articles in articles_by_category.items():
#     for article in articles:
#         headline = article['title']
#         summary = summarize_headline_5_words(headline)

#         # Search for related articles
#         url = "https://newsapi.org/v2/everything"
#         params = {"q": summary, "apiKey": newsapi_key}
#         response = requests.get(url, params=params)
#         related_articles = response.json().get('articles', [])

#         # Create a 75-word summary
#         if related_articles:
#             descriptions = ' '.join([a['description'] for a in related_articles if a['description']])
#             topic_prompt = f"Summarize the following into a 75-word summary:\n\n{descriptions}"
#             perspective_prompt = "Identify and summarize the two major perspectives with specifics in 50 words each based on the following content:\n\n" + descriptions

#             # Summarize topic
#             topic_summary = openai.ChatCompletion.create(
#                 model="gpt-3.5-turbo",
#                 messages=[{"role": "user", "content": topic_prompt}]
#             )['choices'][0]['message']['content'].strip()

#             # Summarize perspectives
#             perspective_summary = openai.ChatCompletion.create(
#                 model="gpt-3.5-turbo",
#                 messages=[{"role": "user", "content": perspective_prompt}]
#             )['choices'][0]['message']['content'].strip()

#             article['summary'] = summary
#             article['topic_summary'] = topic_summary
#             article['perspective_summary'] = perspective_summary

#             print(f"Summary: {summary}")
#             print(f"Topic Summary: {topic_summary}")
#             print(f"Perspective Summary:\n{perspective_summary}\n")