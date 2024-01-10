import os
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

class MarketAnalysis:
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_cities(state, min_pop, max_pop):
        formatted_state = state.lower().replace(" ", "")
        url = f"https://www.citypopulation.de/en/usa/cities/{formatted_state}/"
        
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'ts'})
        
        # Get list of cities in the state
        cities_data = []
        for row in table.tbody.find_all("tr"):
            cells = row.find_all("td")
            city_name = cells[1].text.strip()
            city_population = int(cells[7].text.strip().replace(",", ""))
            cities_data.append(
                {"name": city_name, "population": city_population})
        
        # Get list of cities in the state with populations between min_pop and max_pop
        cities = []
        
        for city in cities_data:
            if city.get("population") >= int(min_pop) and city.get("population") <= int(max_pop):
                cities.append(city.get("name"))
        
        return cities
    
    # Get search results from SerpApi
    @staticmethod
    def get_search_results(niche, location):

        params = {
            "engine": "google",
            "q": niche,
            "location": location,
            "api_key": os.getenv("SERPAPI_API_KEY")
        }

        # Get search results and pull out map pack and organic results
        search = GoogleSearch(params)
        results = search.get_dict()
        # print(results)

        # Get array of places in map pack
        map_pack = results.get("local_results", {}).get("places")
        # print(map_pack)
        
        # Get array of organic results
        organic_results = results.get("organic_results")
        # print(organic_results)

        results = {
            'map_pack': map_pack,
            'organic_results': organic_results
        }

        return results
    
    # Filter map pack
    @staticmethod
    def process_map_pack(map_pack):
        
        filtered_results = []
        
        for result in map_pack:
            filtered_result = {
                'title': result.get('title').lower(),
                'rating': result.get('rating'),
                'reviews': result.get('reviews'),
                'type': result.get('type'),
                'website': result.get('links', {}).get('website')
            }
            filtered_results.append(filtered_result)
            
        return filtered_results

    # Filter organic results
    @staticmethod
    def process_organic_results(organic_results):
        filtered_results = []

        for result in organic_results:
            filtered_result = {
                'title': result.get('title').lower(),
                'link': result.get('link'),
                'description': result.get('about_this_result', {}).get('source', {}).get('description')
            }
            filtered_results.append(filtered_result)

        return filtered_results

    @staticmethod
    def analyze_map_pack(map_pack, city):

        city = city.lower()
        
        # Count instances of parameters: city name in title, more than 10 reviews, and connected websites
        city_in_title = 0
        more_than_10_reviews = 0
        connected_websites = 0
        
        for result in map_pack:
            # Check if all words in the city name are in the title
            if all(word in result.get('title') for word in city.split()):
                city_in_title += 1
            if result.get('website'):
                connected_websites += 1
            if result.get('reviews'):
                if result.get('reviews') > 10:
                    more_than_10_reviews += 1
        
        results = {
            'city_in_title': city_in_title,
            'more_than_10_reviews': more_than_10_reviews,
            'connected_websites': connected_websites
        }
        
        return results
    
    @staticmethod
    def analyze_organic_results(organic_results, city):
        
        city.lower()
        
        # Count instances of parameters: city name in title, city name in link
        city_in_title = 0
        city_in_link = 0
        
        for result in organic_results:
            # Check if all words in the city name are in the title or link
            if all(word in result.get('title') for word in city.split()):
                city_in_title += 1
            if all(word in result.get('link') for word in city.split()):
                city_in_link += 1
        
        results = {
            'city_in_title': city_in_title,
            'city_in_link': city_in_link,
        }
        
        return results

    @staticmethod
    def analyze_types(map_pack, niche):
        
        types = []
        
        for result in map_pack:
            types.append(result.get('type'))
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)
        
        prompt = PromptTemplate(
            input_variables=["types", "niche"],
            template="""
            Types: {types}
            Count how many 'types' represent a business that would offer {niche} services.
            
            Respond with only the number of business types that match {niche}.
            example response: '2'
            DO NOT RESPOND IN A SENTENCE FORMAT. ONLY RESPOND WITH THE NUMBER.
            """
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        response = chain.run(types=types, niche=niche)
        
        return response
    
    @staticmethod
    def analyze_descriptions(organic_results, niche):
        
        descriptions = []
        
        for result in organic_results:
            descriptions.append(result.get('description'))
            
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)
        
        prompt = PromptTemplate(
            input_variables=["descriptions", "niche"],
            template="""
            Descriptions: {descriptions}
            Count how many 'descriptions' are 'local sites', AKA this website is a business offering {niche} services in the area.
            
            Local sites have descriptions like this: "___ was first indexed by Google in May 2015", as they are not well versed on SEO and don't have a description of their business. Other times they have: "___ is a landscaping company that offers landscaping services in the area. We are located in the city of 'City Name' and have been in business since 2015."
            
            Respond with only the number of sites that are 'Local'.
            Example response: '3'
            DO NOT RESPOND INA SENTENCE FORMAT. ONLY RESPOND WITH THE NUMBER.
            IF YOU DONT KNOW REPLY WITH '0'
            """
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        response = chain.run(descriptions=descriptions, niche=niche)
        
        return response
    
    @staticmethod
    def prepare_response(map_pack_analysis, organic_analysis, type_analysis, description_analysis):
        
        header = {
            'type': 'section',
            'text': {
                'type': 'mrkdwn',
                'text': '*Competition Analysis Results*'
            }
        }
        
        divider = {'type': 'divider'}
        
        map_pack_table = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Map Pack Results (max: 3 competitors):*\n"
                    f"City in title: {map_pack_analysis['city_in_title']} / 3\n"
                    f"More than 10 reviews: {map_pack_analysis['more_than_10_reviews']} / 3\n"
                    f"Connected websites: {map_pack_analysis['connected_websites']} / 3\n"
                    f"Relevent types: {type_analysis} / 3\n"
                )
            }
        }

        organic_results_table = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Organic Results (max: 10 competitors):*\n"
                    f"City in title: {organic_analysis['city_in_title']} / 10\n"
                    f"City in link: {organic_analysis['city_in_link']} / 10\n"
                    f"Local websites: {description_analysis} / 10\n"
                )
            }
        }

        blocks = [header, divider, map_pack_table, divider, organic_results_table, divider]

        return blocks
    
    @staticmethod
    def decide_to_proceed(map_pack_analysis, organic_analysis, type_analysis, description_analysis):

        def calculate_map_pack_score():
            # Weights: city_in_title:more_than_10_reviews:connected_websites:relevant_types = 4:3:1:1
            title_weight = 4
            reviews_weight = 3
            websites_weight = 1
            types_weight = 1

            title_value = map_pack_analysis['city_in_title'] * title_weight # max: 12
            reviews_value = map_pack_analysis['more_than_10_reviews'] * reviews_weight # max: 9
            websites_value = map_pack_analysis['connected_websites'] * websites_weight # max: 3
            types_value = type_analysis * types_weight # max: 3

            # Get score out of 100
            score = ((title_value + reviews_value + websites_value + types_value) / 27) * 100
            return score
            
        def calculate_organic_score():
            # Weights: city_in_title:city_in_link:local_sites = 2:3:5
            title_weight = 2
            link_weight = 3
            sites_weight = 5

            title_value = organic_analysis['city_in_title'] * title_weight
            link_value = organic_analysis['city_in_link'] * link_weight
            sites_value = description_analysis * sites_weight

            return title_value + link_value + sites_value

        map_pack_score = calculate_map_pack_score()
        organic_score = calculate_organic_score()

        # Get the total competition scores
        map_pack_weight = 0.80
        organic_score_weight = 0.20

        total_competition_score = (map_pack_score * map_pack_weight) + (organic_score * organic_score_weight)

        # Construct the response
        report_scores = (f"Map Pack Score: {int(map_pack_score)} / 100\n")

        if total_competition_score < 30:
            return report_scores + "Proceed with the campaign in"
        else:
            return report_scores + "Do not proceed with the campaign in"
