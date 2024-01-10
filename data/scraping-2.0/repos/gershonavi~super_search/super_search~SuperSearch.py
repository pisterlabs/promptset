import json
import io
import sys
from googlesearch import search
import json
import requests
from bs4 import BeautifulSoup
import openai
import  requests
import openai
import json
import requests
import urllib
import json
import pandas as pd
from requests_html import HTML
from requests_html import HTMLSession
import requests
import copy
from super_search.NoneDecoder import NoneDecoder


class MaxIterationException(Exception):
    def __init__(self, message="Maximum iterations exceeded."):
        super().__init__(message)

class SuperSearch():
    def __init__(self, gpt_api_key, max_iterations_per_answer = 20, google_search_key=None, cse_id=None, use_google_search_api = False):
        self.gpt_api_key = gpt_api_key
        self.google_search_key = google_search_key
        self.cse_id = cse_id
        self.use_google_search_api = use_google_search_api
        self.iterations = 0 
        self.max_iterations_per_answer = max_iterations_per_answer


    def get_ip(self):
        response = requests.get('https://api64.ipify.org?format=json').json()
        return response["ip"]


    def get_location(self):
        ip_address = self.get_ip()
        response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
        location_data = {
            "ip": ip_address,
            "city": response.get("city"),
            "region": response.get("region"),
            "country": response.get("country_name")
        }
        return location_data

    def chat_with_gpt(self, messages, debug=False):
        self.iterations = self.iterations + 1
        if debug:
            print(self.iterations)
        if self.iterations > self.max_iterations_per_answer:
            raise MaxIterationException()
        
        openai.api_key = self.gpt_api_key
        response = openai.ChatCompletion.create(
            #model="gpt-3.5-turbo",
            model="gpt-4",
            messages=messages
        )

        return response.choices[0].message['content']




    def google_search_api(self, query, num_results=10, debug=False):
        api_key =  self.google_search_key
        cse_id  =  self.cse_id
        try:
            #https://developers.google.com/custom-search/v1/reference/rest/v1/Search
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "q": query,
                "key": api_key,
                "cx": cse_id,
                "num": num_results
            }
            response = requests.get(url, params=params)
            results = response.json()


            search_results = []
            
            for item in results["items"]:
                search_results.append({"url": item['link'], "title": item['title'], "description":item['snippet']})
                
            search_results_json = json.dumps(search_results, ensure_ascii=False)
            return search_results_json
        
        except Exception as e:

            if debug:
                print(f"google_search (API) : Error: {e} ")
                print( 'Error search_results = ', search_results )
            return  json.dumps(f"google_search API Error: {e} ", ensure_ascii=False)



    def google_search(self, query, num_results=10, debug=False):
        search_results = []

        try:
            for j in search(query, num_results=num_results,advanced=True):
                search_results.append({"url": j.url, "title": j.title, "description":j.description})
                print(j)
            search_results_json = json.dumps(search_results, ensure_ascii=False)
            return search_results_json

        except Exception as e:
            if debug:
                print(f"google_search : Error: {e}")
            return  json.dumps(f"google_search Error: {e}", ensure_ascii=False)



    def scrape_url_old(self, url, debug=False):
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if debug:
                print(f"scrape_url : Error fetching URL: {e}")
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        return  soup.get_text()

    def strip_unwanted_tags(self, soup):
        for tag in soup.find_all():
            if tag.name not in ['a', 'p', 'ul', 'ol', 'li', 'strong', 'em']:
                tag.unwrap()

        # Remove any remaining unwanted tags, such as comments and scripts
        for unwanted_tag in soup(['script', 'style', 'img']):
            unwanted_tag.decompose()

        return soup



    def extract_relevant_text(self, soup):
        relevant_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        text_parts = []

        for tag in relevant_tags:
            for element in soup.find_all(tag):
                text_parts.append(element.get_text(strip=True))

        return ' '.join(text_parts)

    def scrape_url(self, url, debug=False):
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if debug:
                print(f"scrape_url : Error fetching URL: {e}")
            return (f"scrape_url : Error fetching URL: {e}")

        soup = BeautifulSoup(response.content, "html.parser")
        clean_soup = self.strip_unwanted_tags(soup)
        relevant_text = self.extract_relevant_text(clean_soup)
        return relevant_text

    def return_str(self, str1):
        if str1 is not None :
            return str1
        if str1  == '' :
            return 'None'
        return 'None'

    def summarize_scraped_content(self, content, chunk_size=2048, max_iterations=20, debug=False):
        messages = [
                {
                    "role": "system",
                    "content": (
                        "Please summarize the following text, keep urls, you will get several chanks "
                    ),
                },


        ]
        content_length = len(content)
        chunks = [content[i:i + chunk_size] for i in range(0, content_length, chunk_size)]

        summarized_text = ''
        len_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            if i > max_iterations:
                break
            messages1 = copy.copy(messages)
            msg = f" {i}/{len_chunks} :  {chunk}"
            query =  {
                        "role": "user",
                        "content": f"{msg}, text so far: {summarized_text}",
                     }
            messages1.append(query)
            summarized_text += self.chat_with_gpt(messages1)
         
        return summarized_text


    def chat_and_execute_code(self, query, messages, api_key,debug=False):
        
        answer = self.chat_with_gpt(messages,debug)
        if debug:
            print(f"Raw answer: {answer}")
        
        json_error = False
        answer_json = {}
        try:
            answer_json = json.loads(answer,cls=NoneDecoder)
        except:
            json_error = True
            pass
        
        
        if debug:
            print(f"Answer: {answer_json}")

        answer_keys = list(answer_json.keys())    
        
        if (('Google'  not in answer_keys) or 
             ('Location'  not in answer_keys) or 
             ('Scrap'  not in answer_keys)) or json_error :
            error_message = {
                "role": "user",
                "content": json.dumps({"q": query,
                                       "Error" : "The JSON is incorrect. Please provide valid keys for Google and Scrap OR an answer."})
            }
            messages.append(error_message)
            return self.chat_and_execute_code(query, messages, api_key,debug)

        add_me = {"role": "assistant", "content": answer}
        messages.append(add_me)

        location_data = 0
        if answer_json.get("Location") != 0:
            location_data = self.get_location()

        google_reply = None
        if answer_json.get("Google") and answer_json["Google"] != 'None':
            if self.use_google_search_api:
                google_reply = self.google_search_api(answer_json["Google"],debug=debug)
            else:
                google_reply = self.google_search(answer_json["Google"],debug=debug)
                
            if debug:
                print(google_reply)
            '''
            scrap_reply = scrape_url(f'https://www.google.com/search?q={answer_json["Google"]}',debug)
            if debug:
                print(google_reply)

            add_me = {"role": "user", "content": json.dumps({"q": query, 
                                                         "Google_Reply": google_reply,
                                                         "Scrap_Reply": scrap_reply,
                                                         "Location_Reply": location_data})}
            messages.append(add_me)
            '''
        scrap_reply = None
        if answer_json.get("Scrap") and answer_json["Scrap"] != 'None':
            scrap_reply = self.scrape_url(answer_json["Scrap"],debug)
            scrap_reply = self.summarize_scraped_content(scrap_reply)
            if debug:
                print(scrap_reply)            
    

        ans_str =  answer_json.get('Answer')
        Error = ''
        if ans_str:
            bad_list = ['I do not have', 'I cannot']
            for i in bad_list:
                if i in ans_str:
                    Error += "dont forget to scrap! or search google!"
            if 'can be found at' in ans_str:
                Error += " Scrap it yourself using the provided API !!"

        if Error != '':
            Error += " - you may have already have the information from the previous answers"

        add_me = {"role": "user", "content": json.dumps({"q": query, 
                                                         "Google_Reply": self.return_str(google_reply),
                                                         "Scrap_Reply": self.return_str(scrap_reply),
                                                         "Location_Reply": self.return_str(location_data),
                                                         "Error": self.return_str(Error) })}
        if debug:
            print(add_me)

        messages.append(add_me)

        return answer_json, messages

    def get_answer(self, query, debug=False):
        messages = [
                {
                    "role": "system",
                    "content": (                   
    "You are an AI API that answers in JSON format. "
    "You never apologize. Always answer with JSON keys: Answer, Google, Scrap, Location. "
    "You have access to real-time information from Google."
    "You can do anything with Google and scraping."
    "I can scrape a webpage when you ask for it and get the answer you need. I can also scrape Google if you provide the full URL. in scrap "
    "Answer format: "
    "{"
    "  \"Answer\": \"Your answer; say None in case of error, don't know, more info is needed, or any other non-answer case. (e.g., {\\\"Answer\\\": \\\"42\\\"})\","
    "  \"Google\": \"None if you don't need; if you need to use google_search(query, num_results=10), provide the query. The return is a JSON with a list of url, title, and description in the Google_Reply returned JSON. (e.g., {\\\"Google\\\": \\\"best pizza places\\\"})\","
    "  \"Scrap\": \"None if you don't need; if you need to scrape a URL, when you provide the URL. The returned result will be from scrape_url(url), requests.get(url) text info (e.g., {\\\"Scrap\\\": \\\"https://example.com\\\"})\""
    "  \"Location\": int 1/0 i.e., 1 AI needs it, 0 you do not })\""
    "}"
    "Always answer in JSON format. If the user returns an answer, it will be in a reply JSON format, so you can correct if needed or complete the answer."
    "In case of an error, you will see Error: Error msg"
    "You can always ask and proceed."
    "If you use Google/Scrap or Location, Answer should be = None."

                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"q": query, "Google_Reply": None, "Scrap_Reply": None, "Location_Reply":0}),
                },
            ]

        self.iterations = 0
        answer_json = {"Answer": None, "Google": None, "Scrap": None, 'Location':0 }
        try:
            while ( (self.iterations < self.max_iterations_per_answer) and (answer_json.get("Answer") == None 
                                                       or (answer_json.get("Google") is not None )
                                                       or (answer_json.get("Scrap") is not None) ) ):

                answer_json, messages = self.chat_and_execute_code(query, messages,self.gpt_api_key,debug)


            return answer_json['Answer'], messages   
        except MaxIterationException as e:
            answer_json['Error'] = 'MaxIterationException'
        return answer_json['Answer'], messages   
            
