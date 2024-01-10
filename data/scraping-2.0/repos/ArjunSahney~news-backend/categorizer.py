import requests
import openai

# API Keys
news_api_key = '785379e735a84282af1c6b35cf335a59'
openai.api_key = "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"

# Fetch articles from NewsAPI
def fetch_articles_from_source(source, num_articles=100):
    articles_list = []
    pages_needed = (num_articles + 99) // 100
    base_url = 'https://newsapi.org/v2/everything'
    headers = {"Authorization": f"Bearer {news_api_key}"}

    for page in range(1, pages_needed + 1):
        params = {
            "from": "2023-10-10",
            "sortBy": "popularity",
            "sources": source,
            "pageSize": 100,
            "page": page,
            "language": 'en'
        }

        response = requests.get(base_url, headers=headers, params=params)
        data = response.json()
        articles_list.extend(data.get('articles', []))

    return articles_list[:num_articles]

# Categorization using Zero-shot approach
def generate_text_labels(texts):
    categories = ["news", "business", "technology", "sports", "entertainment"]
    labels = []
    text_label_mapping = {}
    category_str = ", ".join(categories)

    for text in texts:
        response = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=[
                        {"role": "user", "content": f"{text}; Classify this sentence as {category_str} in one word."}
                    ]
                )
        label = response.choices[0]["message"]["content"].strip(".")
        labels.append(label)
        text_label_mapping[text] = label

    return labels, text_label_mapping

# Fetch and store articles by category and source
def fetch_and_store_articles_by_category_and_source():
    sources_by_category = {
        "news": ["google-news"],
        "business": ["business-insider"],
        "technology": ["techcrunch", "the-verge"]
    }

    num_articles_by_category = {
        "news": 200,
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

# Display articles and their categories
def display_articles_and_categories():
    articles = fetch_and_store_articles_by_category_and_source()
    titles = [article['title'] for category, articles_list in articles.items() for article in articles_list]
    _, mapping = generate_text_labels(titles)

    print(mapping)

    for title, category in mapping.items():
        print(f"Title: {title}\nCategory: {category}\n")

# Driver code
if __name__ == '__main__':
    display_articles_and_categories()


### Has dictionary that you get of categories and headlines from each. Aim is to use NLP to summarize headline for keywords (4 word summary), search keywords using news api, scrape news api using llm, summarize different stories
