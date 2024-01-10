from httpx import HTTPError
import openai
import re
import nltk
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from collections import Counter
from sklearn.cluster import KMeans
from gensim.matutils import corpus2dense
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import TfidfModel, LdaModel
from api.models.analysis import Analysis
from api.models.search_analysis_rel import SearchAnalysisRel
from api.models.search_query import SearchQuery

from api.assets import constants
from api.utils.db import add_commit_
from config import Config
from api.utils import logging_wrapper

logger = logging_wrapper.Logger(__name__)

class CompetitorUtils:
    def get_data(project_id):
        searches = (
            SearchQuery.query.filter(
                SearchQuery.seo_project_id == project_id,
                SearchQuery.type == constants.ProjectTypeCons.enum_competitor,
            ).all()
        )

        searches_ids = [search.id for search in searches]

        analysis_data = (
            SearchAnalysisRel.query.filter(
                SearchAnalysisRel.search_query_id.in_(searches_ids),
            )
        )

        analysis_ids = [analysis.analysis_id for analysis in analysis_data]

        analysis_data = (
            Analysis.query.filter(
                Analysis.id.in_(analysis_ids),
                Analysis.type == constants.ProjectTypeCons.enum_competitor,
            ).all()
        )

        comptitor_analysis = []

        for analysis in analysis_data:
            response = {
                "id": analysis.id,
                "title": analysis.title,
                "snippet": analysis.snippet,
                "link":analysis.link,
                "displayLink": analysis.display_link,
                "htmlSnippet": analysis.html_snippet,
                "htmlTitle": analysis.html_title,
                "pagemap":analysis.pagemap,
                "kind":analysis.kind,
                "htmlFormattedUrl": analysis.html_formatted_url,
                "formattedUrl": analysis.formatted_url,
            }

            comptitor_analysis.append(response)

        return comptitor_analysis

    def generate_competitor_search_text_gpt4(user_id, business_type, target_audience, industry, location):
        try:
            system_prompt = {
                "role": "system",
                "content": "You are a Competitor finder assistant. Your primary work is to write search queries based on user input that provides relevant results which will be helpful for users.\n\nYour response should be pointwise."
            }

            user_prompt = {
                "role": "user",
                "content": f"User Input\n```\nBusiness type: {business_type}\nTarget audience: {target_audience}\nIndustry: {industry}\nLocation: {location}\n```"
            }

            assistant_response = openai.ChatCompletion.create(
                model=Config.OPENAI_MODEL_GPT4,
                messages=[system_prompt, user_prompt],
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                user=str(user_id),
            )

            # Extract and format search queries as an array
            search_queries = assistant_response["choices"][0]["message"]["content"].strip().split("\n")
            total_tokens = assistant_response['usage']['total_tokens']
            return search_queries, total_tokens
        except Exception as e:
            logger.exception(str(e))
            return None, 0

    
    def fetch_competitors(query_arr, project_id, country_code, max_results=5):
        total_point = 0.0
        competitor_urls = {}
        url = "https://www.googleapis.com/customsearch/v1"
            
        for query in query_arr:
            try:
                search_model = SearchQuery.query.filter(
                    SearchQuery.search_query == query,
                    SearchQuery.seo_project_id == project_id,
                    SearchQuery.type == constants.ProjectTypeCons.enum_competitor,
                ).first()

                if search_model is None:
                    search_model = SearchQuery(
                        search_query=query,
                        seo_project_id=project_id,
                        type=constants.ProjectTypeCons.enum_competitor,
                    )
                    add_commit_(search_model)

                params = {
                    "key": Config.GOOGLE_SEARCH_API_KEY,
                    "cx": Config.CUSTOM_SEARCH_ENGINE_ID,
                    "q": query,
                    'num': max_results,
                    'gl': country_code,
                }

                response = requests.get(url, params=params)
                if response.status_code == 200:
                    competitor_urls[search_model.id] = response.json().get("items", [])
                else:
                    continue
            except HTTPError as e:
                logger.exception(str(e))
                continue

        total_point = len(competitor_urls) * 0.0005
        return competitor_urls, total_point

    # Preprocessing function
    def preprocess(text):
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = text.lower()  # Convert text to lowercase
        tokens = nltk.word_tokenize(text)  # Tokenize words
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stopwords.words('english')]
        return tokens

    def keywords_titles_builder(news_data):
        # Preprocess the articles
        articles = [(article.get("title", "") + article.get("snippet", "")) for article in news_data]
        preprocessed_articles = [CompetitorUtils.preprocess(article) for article in articles]

        # Build a dictionary and a corpus for the articles
        dictionary = Dictionary(preprocessed_articles)
        corpus = [dictionary.doc2bow(article) for article in preprocessed_articles]

        # Calculate TF-IDF scores
        tfidf_model = TfidfModel(corpus)
        tfidf_corpus = tfidf_model[corpus]

        # Convert the tfidf_corpus to a dense matrix
        dense_tfidf_corpus = corpus2dense(tfidf_corpus, num_terms=len(dictionary)).T


        # Extract top keywords and key phrases
        top_keywords = Counter()
        for doc in tfidf_corpus:
            top_keywords.update({dictionary[word_id]: score for word_id, score in doc})

        # Topic modeling using LDA
        lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)
        
        # Compute the topic coherence score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=preprocessed_articles, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()

        # Filter out irrelevant topics based on a minimum threshold (e.g., 0.3)
        min_coherence_threshold = 0.3
        if coherence_lda >= min_coherence_threshold:
            topics = lda_model.show_topics(num_topics=5, num_words=5, formatted=False)
        else:
            # Handle the case where the topics are not coherent enough
            print("The topics are not coherent enough. Please try again later.")

        # Clustering using KMeans
        n_clusters = min(5, len(dense_tfidf_corpus))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        clusters = kmeans.fit_predict(dense_tfidf_corpus)

        # Analyzing trends and events
        trending_topics = []
        for cluster in range(5):
            cluster_articles = [article for article, c in zip(articles, clusters) if c == cluster]
            cluster_keywords = Counter()
            for article in cluster_articles:
                article_keywords = CompetitorUtils.preprocess(article)
                cluster_keywords.update(article_keywords)
            trending_topics.append(cluster_keywords.most_common(5))

        return trending_topics

    def generate_title(user_id, keywords, app):
        with app.app_context():
            # Join the top keywords with a comma
            keywords_str = ", ".join([keyword[0] for keyword in keywords])

            system_prompt = {
                "role": "system",
                "content": "You are an AI assistant trained to generate relevant and engaging article titles based on a set of keywords. Generate a title using the following keywords."
            }

            user_prompt = {
                "role": "user",
                "content": f"Keywords: {keywords_str}"
            }

            assistant_response = openai.ChatCompletion.create(
                model=Config.OPENAI_MODEL_GPT4,
                messages=[system_prompt, user_prompt],
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                user=str(user_id),
            )

            # Extract and format the title
            title = assistant_response["choices"][0]["message"]["content"].strip()
            total_tokens = assistant_response['usage']['total_tokens']
            return title, total_tokens