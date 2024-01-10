import openai
import requests
import spacy
import re
import multiprocessing
from transformers import pipeline
from clickbait import clickbait
from spellchecker import SpellChecker
from subjectivemodel import subjective
from urllib.parse import urlparse
from newspaper import Config
import nltk
from nltk.tokenize import sent_tokenize
from gingerit.gingerit import GingerIt
from isnewstitle import checkNewsTitle
from bs4 import BeautifulSoup
from similarity import calculate_sentence_similarity
import time
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import html5lib
from selenium import webdriver

config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

API_KEY = "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"
openai.api_key = API_KEY


class checkTitle:

    def __init__(self, title):
        self.headline = title
        self.own_corrections = {'iam': "i'm", 'im': "i'm"}
        self.spellings_ratio = 0
        self.corrected = title
        self.misspelled_words = []
        self.required = []
        self.contexts = []

    def lower_case(self, text):
        text = text.lower()
        pattern = re.compile("<.*?>")
        text = pattern.sub(r'', text)
        pattern = re.compile(r'https?://\S+|www\.\S+')
        text = pattern.sub(r'', text)
        exclude = "[!\#\$%\&\(\)\*\+,\.\-\"/:;<=>\?@\[\]\^_`\{\|\}\~0123456789]"
        return text.translate(str.maketrans('', '', exclude))

    def spelling_mistakes(self):
        head = self.lower_case(self.headline)
        output_dir = "C://Users/vixha/downloads"
        ner_pipeline = pipeline("ner", grouped_entities=True, model=output_dir)
        head2 = ' '.join([i.capitalize() for i in head.split()])

        ner_results = ner_pipeline(head2)
        named_entities = [entity["word"].lower() for entity in ner_results if entity["entity_group"] == "PER" or entity["entity_group"] == "LOC"]

        misspelled_words = []
        # parser = GingerIt()
        spell = SpellChecker()
        words = []

        for token_text in head.split(' '):
            # corrected_token = parser.parse(token_text)['result'].lower()
            corrected_token = spell.correction(token_text)
            if token_text.isalpha() and token_text not in named_entities:
                if token_text in self.own_corrections:
                    words.append(self.own_corrections[token_text])
                    misspelled_words.append(token_text)
                elif token_text != corrected_token:
                    misspelled_words.append(token_text)
                    words.append(corrected_token)
                else:
                    words.append(token_text)
            else:
                words.append(token_text)

        self.corrected = ' '.join(words)
        self.misspelled_words = set(misspelled_words)
        if len(misspelled_words) == 0:
            return True
        ratio = len(misspelled_words) / len(self.headline.split(" "))
        return ratio < 0.5

    def classify_clickbait(self):
        click = clickbait(self.corrected)
        return click.run() == 0

    def subjective_test(self):
        subjective_obj = subjective()
        answer = subjective_obj.send_request(self.headline)
        return answer == "objective"

    def is_newstitle(self):

        if len(self.headline) > 90:
            return False

        if not re.search(r'[A-Z][a-z]+', self.headline):
            return False

        is_news = checkNewsTitle(self.headline).run()
        if is_news[0] == 0:
            return False
        return True

    def present_on_google(self):
        url = f"https://www.google.com/search?q={self.headline}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        include_link = soup.find('a', class_='fl')
        if include_link:
            response = requests.get('https://www.google.com/' + include_link['href'], headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

        search_results = soup.find_all('div', class_='yuRUbf')
        search_contexts = soup.find_all('div', {'class': ['VwiC3b', 'yXK7lf', 'MUxGbd', 'yDYNvb', 'lyLwlc', 'lEBKkf']})
        search_contexts = [i for i in search_contexts if
                           str(i).find('class="VwiC3b yXK7lf MUxGbd yDYNvb lyLwlc lEBKkf"') != -1]
        urls = []
        for result, context in zip(search_results, search_contexts):
            link = result.find('a')
            url = link['href']
            heading = result.find(re.compile('^h[1-6]$')).text
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            parsed_url = urlparse(url)
            domain_name = parsed_url.netloc
            domain_name = domain_name.replace('www.', '')
            if soup.find('title') and self.present_on_google_news_2(domain_name):

                self.required.append(soup.find('title').text)
                if len(context.find_all('span')) > 2:
                    self.contexts.append(context.find_all('span')[2].text)
                elif len(context.find_all('span')) > 1:
                    self.contexts.append(context.find_all('span')[1].text)
                else:
                    self.contexts.append(context.find_all('span')[0].text)
                urls.append({'url': url, 'heading': heading})

        print(self.contexts)

        if len(self.required) < 3:
            return False
        return self.availability_on_web(urls)

    def availability_on_web(self, results):
        similar_links = []
        max_similarity = 0
        article_heading = ''

        for result, context in zip(self.required, self.contexts):
            similarity_percentage_1 = calculate_sentence_similarity(self.headline, result)
            print(self.headline, "   ", result, "   ", similarity_percentage_1)
            similarity_percentage_2 = calculate_sentence_similarity(self.headline, context)
            print(self.headline, "   ", context, "  2 ", similarity_percentage_2)
            if similarity_percentage_1 > similarity_percentage_2:
                if similarity_percentage_1 > max_similarity:
                    article_heading = result
            else:
                if similarity_percentage_2 > max_similarity:
                    article_heading = context
            similarity_percentage = max(similarity_percentage_1, similarity_percentage_2)
            if similarity_percentage >= 0.55:
                similar_links.append(similarity_percentage)

        if len(similar_links) < 2:
            return False

        if not self.check_similarity2(article_heading):
            return False

        return True

    def present_on_google_news_2(self, domain):
        print(domain)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        ggl_news_link = f"https://www.google.com/search?q={domain}&tbm=nws"
        req = requests.get(ggl_news_link, headers=headers)
        sup = BeautifulSoup(req.content, 'html.parser')
        link = sup.find('a', class_='WlydOe')
        if link:
            nd_domain = urlparse(link['href'])
            domain_name = nd_domain.netloc
            domain_name = domain_name.replace('www.', '')
            print(domain, domain_name)
            if domain == domain_name:
                return True
        return False

    def check_similarity2(self, context):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"i have two sentences \nsentence1 = {self.headline} \nsentence2 = {context} \ndont consider additional information, based on the contextual similarity, is the first statement true based on second statement yes or no",
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response_text = response.to_dict()['choices'][0]['text'].replace("\n", "")
        print(response_text)
        final_response = response_text[:4]
        if final_response.lower().find('yes') != -1:
            return True
        else:
            return False

    def run(self):
        print(self.is_newstitle())


checkTitle("hi").present_on_google_news_2('hello how are you')
# if __name__ == '__main__':
#     x = checkTitle("DRDO and L&T will Sign Contract To Develop 'Indigenous' AIP System For indian Navy Submarines")
#     tim = time.time()

#     p1 = multiprocessing.Process(target=x.spelling_mistakes)
#     p2 = multiprocessing.Process(target=x.classify_clickbait)
#     p3 = multiprocessing.Process(target=x.subjective_test)
#     p4 = multiprocessing.Process(target=x.is_newstitle)
#     p5 = multiprocessing.Process(target=x.present_on_google)

#     p1.start()
#     p2.start()
#     p3.start()
#     p4.start()
#     p5.start()

#     p1.join()
#     p2.join()
#     p3.join()
#     p4.join()
#     p5.join()

#     print(time.time() - tim)
