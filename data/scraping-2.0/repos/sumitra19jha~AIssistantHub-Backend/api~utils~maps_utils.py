import openai
import yake
from googleplaces import GooglePlaces
from api.models.analysis import Analysis
from api.models.search_analysis_rel import SearchAnalysisRel
from api.models.search_query import SearchQuery
from config import Config
from api.utils import logging_wrapper

import requests
from bs4 import BeautifulSoup
import spacy
import re

import numpy as np
from sklearn.cluster import KMeans
from api.assets import constants


logger = logging_wrapper.Logger(__name__)
nlp = spacy.load("en_core_web_sm")

class AssistantHubMapsAlgo:
    def get_data(project_id):
        searches = (
            SearchQuery.query.filter(
                SearchQuery.seo_project_id == project_id,
                SearchQuery.type == constants.ProjectTypeCons.enum_maps,
            ).all()
        )

        searches_ids = [search.id for search in searches]

        analysis_data = (
            SearchAnalysisRel.query.filter(
                SearchAnalysisRel.search_query_id.in_(searches_ids),
            )
        )

        analysis_ids = [analysis.analysis_id for analysis in analysis_data]
        print(analysis_ids)

        analysis_data = (
            Analysis.query.filter(
                Analysis.id.in_(analysis_ids),
                Analysis.type == constants.ProjectTypeCons.enum_maps,
            ).all()
        )

        maps_data = []

        for analysis in analysis_data:
            response = {
                "id": analysis.id,
                "title": analysis.title,
                "address": analysis.address,
                "google_maps_url": analysis.map_url,
                "name": analysis.name,
                "optimized_snippets": analysis.snippet,
                "website": analysis.website,
                "backlinks": analysis.backlinks,
                "keywords": analysis.keywords,
                "latitude": analysis.latitude,
                "longitude": analysis.longitude,
            }

            maps_data.append(response)

        return maps_data

    def analyze_georaphic_distribution(array_of_maps_data):
        if not array_of_maps_data:
            return {}  # Return an empty dictionary if the input is empty

        coordinates = np.array([[place["latitude"], place["longitude"]] for place in array_of_maps_data])
        
        # Granularity of the analysis
        num_clusters = 5

        # Handle the case when there are fewer samples than the desired number of clusters
        if len(coordinates) < num_clusters:
            num_clusters = len(coordinates)

        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coordinates)
        for i, place in enumerate(array_of_maps_data):
            place["cluster_label"] = int(kmeans.labels_[i]) 

        cluster_summary = {}
        for place in array_of_maps_data:
            label = place["cluster_label"]
            if label not in cluster_summary:
                cluster_summary[label] = {"count": 0, "places": []}
            cluster_summary[label]["count"] += 1 
            cluster_summary[label]["places"].append(place["name"])

        return cluster_summary

    # Fetch News search text
    def generate_maps_search_text_gpt4(user_id, business_type, target_audience, industry, location):
        try:
            system_prompt = {
                "role": "system",
                "content": "You are an AI assistant trained to generate relevant search query for google maps based on user input. Generate a maps search query using the following inputs."
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

    # Uses Google Places API for search based on user Input
    # The current form of Query for Search is:
    #   "{business_type} {target_audience} {industry} {goals}"
    def fetch_google_places(query):
        google_places = GooglePlaces(Config.GOOGLE_SEARCH_API_KEY_FOR_PLACES)
        query_result = google_places.text_search(query=query)

        places_data = []
        for place in query_result.places:
            place.get_details()
            place_dict = {
                "name": place.name,
                "address": place.formatted_address,
                "google_maps_url": f"https://maps.google.com/?q={place.geo_location['lat']},{place.geo_location['lng']}",
                "latitude": place.geo_location["lat"],
                "longitude": place.geo_location["lng"]
            }
            if hasattr(place, 'website'):
                place_dict["website"] = place.website
            if hasattr(place, 'rating'):
                place_dict["rating"] = place.rating
            places_data.append(place_dict)

        return places_data

    def fetch_website_data(url):
        response = requests.get(url)

        # Return None if the status code indicates scraping is not allowed
        if response.status_code == 403:
            return None

        soup = BeautifulSoup(response.content, "html.parser")

        title = soup.title.string if soup.title else ""
        snippets = [snippet.text for snippet in soup.find_all("p")]
        urls = [link.get("href") for link in soup.find_all("a")]
        return title, snippets, urls

    def process_text(text, place, num_keywords=10):
        # Initialize YAKE
        language = "en"
        max_ngram_size = 3
        deduplication_threshold = 0.9
        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold)

        # Extract keywords using YAKE
        keywords = custom_kw_extractor.extract_keywords(text)
        
        # Get the top num_keywords keywords
        top_keywords = [kw[0] for kw in keywords[:num_keywords]]

        # Add localized information to the keywords
        localized_keywords = top_keywords + [place["name"], place["address"]]

        return localized_keywords