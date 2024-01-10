# Author: Ryder Rhoads
# File: Topics.py
# Description: This file defines the SearchTopic class, which facilitates the searching of news articles on 
# specific topics, summarizing these articles using OpenAI's GPT model, and organizing the information. 
# It leverages external APIs and natural language processing techniques to identify, group, and summarize 
# relevant news articles.
import json
import os
from openai import OpenAI
import re
from newspaper import Article
from newspaper import Config
from newsapi import NewsApiClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Initalize API keys and other variables
news_api_key = os.environ.get("NEWSAPI_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


class SearchTopic:
    def __init__(self, topics, similarity_threshold):
        # save topic term and get articles
        self._terms = topics
        self._articles = []
        self.get_articles()
        self._similarity_threshold=similarity_threshold

    def get_articles(self):
        """Function to download articles sorted by time from the newsAPI on a given topic"""
        titles = []
        urls=[]
        newsapi = NewsApiClient(api_key=news_api_key)
        # loop set so we can add more articles if needed
        for term in self._terms:
            api_response = newsapi.get_everything(
                q=term,
                language="en",
                sort_by="publishedAt",
                page_size=100,  # max is 100 allowed by API
            )

            for article in api_response["articles"]:
                # Checks for duplicate articles and if they have a summary
                if article["title"] not in titles and article["url"] not in urls:
                    self._articles.append(
                        {
                            "title": article["title"],
                            "url": article["url"],
                            "image": article["urlToImage"],
                            "imageSource": article["source"]["name"],
                            "text": "",
                        }
                    )
                    titles.append(article["title"])
                    urls.append(article["url"])

            # Download article text and calculate sentiment
            for article_dict in self._articles:
                config = Config()
                config.browser_user_agent = "Mozilla/5.0..."
                article = Article(article_dict["url"], config=config)
                try:
                    article.download()
                    article.parse()
                    article_dict["text"] = article.text
                except Exception as e:
                    print(f"Error downloading article: {e}")
            

    def preprocess_text(self, text):
        """Function to preprocess text for TF-IDF"""
        text = text.lower()
        pattern = r"[^\w\s]"
        preprocessed_text = re.sub(pattern, "", text)
        return preprocessed_text

    def calculate_similarity(self):
        """Convert articles to TF-IDF"""
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(
            [self.preprocess_text(article["text"]) for article in self._articles]
        )
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return cosine_sim

    def find_article_groups(self):
        """Function to find groups of similar articles"""
        similarity_scores = self.calculate_similarity()
        num_articles = len(similarity_scores)
        article_groups = {}
        group_id = 0
        grouped_articles = set()

        for i in range(num_articles):
            if i in grouped_articles:
                continue
            similar_articles = [i]
            for j in range(num_articles):
                if i != j and similarity_scores[i][j] > self._similarity_threshold:
                    similar_articles.append(j)
                    grouped_articles.add(j)

            if len(similar_articles) > 1:
                article_groups[group_id] = similar_articles
                group_id += 1

        return article_groups

    def article_summaries(self, articles):
        """Function to generate summaries for articles using GPT-3.5-turbo-16k"""
        summaries = []
        for article in articles:
            # Prepare the prompt with article title and text
            prompt = (f'Summarize the following article in the most concise way possible, highlighting the main ideas '
                    f'and key points. The summary should also identify any notable bias in a few sentences.\n\n'
                    f'Title: {article["title"]}\n\n'
                    f'Text: {article["text"]}')

            try:
                # API call to OpenAI to generate summary
                response = client.chat.completions.create(model="gpt-3.5-turbo-16k",
                                                        messages=[
                                                            {"role": "system", "content": "You are a helpful assistant."},
                                                            {"role": "user", "content": prompt},
                                                        ])

                # Access the 'choices' attribute of the response object
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    # Extract and append the content of the first choice's message
                    content = response.choices[0].message.content if hasattr(response.choices[0].message, 'content') else ""
                    summaries.append(content.strip())
                else:
                    print("No content in response choices.")

            except Exception as e:
                print(f"Error generating summary: {e}")

        return summaries


    def create_prompt(self, articles):
        """Function to create a prompt for GPT-3.5 that includes summaries of provided articles."""
        prompt = (
            "Another GPT has summarized some articles. Now, create a high-level briefing based on these summaries which will be used as a news story. "
            "You should refer to articles in the plural form and the briefing should be returned in a JSON structure with the following fields: "
            "'title' for a general title, 'background' for an overview of the topic, 'bias' a summary of the bias of the articles, "
            "'summary' for a combined summary of all relevant articles, and 'topics' to list applicable categories. "
            "The categories include: economics, technology, politics, health, business, sports, entertainment, science, world. "
            "The JSON structure should be as follows: {title: String, background: String, summary: String, bias: String, topics: Array}. "
            "Please make sure the topics are only the ones provided and they are in a python list format: ['str2','str1',...] , and you keep the JSON structure the same."
            "Additionally try to avoid referencing the articles directly and if you do refer to them in the plural tense. Below are the summaries:\n\n"
        )
        summaries = self.article_summaries(articles)
        for index, summary in enumerate(summaries):
            prompt += f"Summary {index}:\n{summary}\n\n"
        return prompt

    def export_GPT_summaries(self):
        """Function to export GPT-3 summaries to text files, along with the summaries from the articles"""
        jsons = []
        article_groups = self.find_article_groups()
        for group_id, article_indices in article_groups.items():
            try:
                first_article_idx = article_indices[0]  
                first_article = self._articles[first_article_idx]  
                prompt = self.create_prompt([self._articles[idx] for idx in article_indices])
                summary = self.generate_summary(prompt)
                urls = [self._articles[idx]['url'] for idx in article_indices]
                image = first_article['image']  
                imageSource = first_article['imageSource']
                summary_json = {
                    "GPT_response": json.loads(summary) if summary else None,
                    "urls": urls,
                    "imageURL": image if image else None,
                    "imageSource": imageSource if image else None, 
                }
                jsons.append(summary_json)

            except Exception as e:
                print(f"Error exporting topic group: {e}")

        return jsons


    def generate_summary(self, prompt):
        """Function to generate summaries using GPT-3.5-turbo-16k"""
        try:
            response = client.chat.completions.create(model="gpt-3.5-turbo-16k",
                                                    messages=[
                                                        {"role": "system", "content": "You are a helpful assistant."},
                                                        {"role": "user", "content": prompt},
                                                    ])

            # Access the 'choices' attribute of the response object
            if hasattr(response, 'choices') and len(response.choices) > 0:
                # Extract the content from the first choice's message
                content = response.choices[0].message.content if hasattr(response.choices[0].message, 'content') else ""
                return content.strip()  # Return the content after stripping any leading/trailing whitespace
            else:
                print("No content in response choices.")
                return ""

        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""



