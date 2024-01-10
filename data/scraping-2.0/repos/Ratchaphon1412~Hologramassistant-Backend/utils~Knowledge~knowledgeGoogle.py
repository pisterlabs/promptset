from django.conf import settings
import json
import requests
from openai import OpenAI




class KnowledgeGoogle:
    def __init__(self, googleAPI, googleMapAPI,chatGPTToken):
        self.googleKey = googleAPI
        self.googleMapKey = googleMapAPI
        self.chatGPTToken = chatGPTToken

    def findknowledgeGoogle(self, text):
        endpoint = 'https://kgsearch.googleapis.com/v1/entities:search'
        params = {
            'query': text,
            'limit': 5,
            'indent': True,
            'key': self.googleKey,
            'languages': 'th'
        }
        response = requests.get(endpoint, params=params)
        print(response.text)
        articleList = []
        listData = json.loads(response.text)
        if (listData['itemListElement'] != None):
            if (len(listData['itemListElement']) != 0):
                for diclist in listData['itemListElement']:
                    if ('result' in diclist):
                        if ('detailedDescription' in diclist['result']):
                            if ('articleBody' in diclist['result']['detailedDescription']):
                                articleList.append(
                                    diclist['result']['detailedDescription']['articleBody'])
        else:
            pass

        return articleList

    def nearPlaceRestaurant(self, lat, long):
        print(lat, long)
        endpoint = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat}%2C{long}&radius=1500&type=restaurant&key={self.googleMapKey}&language=th"
        print(endpoint)
        print(self.googleMapKey)
        response = requests.get(endpoint)
        dict_response = json.loads(response.text)
        print(dict_response)
        list_restaurant = []

        for i in range(len(dict_response['results'])):
            # print(dict_response['results'][i]['name'])
            dict_compressrestaurant = {}
            dict_compressrestaurant['name'] = dict_response['results'][i]['name']
            dict_compressrestaurant['geometry'] = dict_response['results'][i]['geometry']
            dict_compressrestaurant['vicinity'] = dict_response['results'][i]['vicinity']
            list_restaurant.append(dict_compressrestaurant)
            dict_compressrestaurant = {}

        return list_restaurant

    def ChatGPT_conversation(self, conversation):
        client = OpenAI(api_key=self.chatGPTToken)

        model_id = 'gpt-3.5-turbo'

        response = client.chat.completions.create(model=model_id,
                                                  messages=conversation)
        # api_usage = response['usage']
        # print('Total token consumed: {0}'.format(api_usage['total_tokens']))
        # stop means complete
        # print(response['choices'][0].finish_reason)
        # print(response['choices'][0].index)
        conversation.append(
            {'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
        return conversation
