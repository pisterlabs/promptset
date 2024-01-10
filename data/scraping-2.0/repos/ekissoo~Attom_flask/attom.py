import datetime
import hashlib
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from ast import literal_eval

import os
import pandas as pd 
# os.environ["GOOGLE_ADS_CONFIGURATION_FILE_PATH"] = "credent.yaml"
# from google.ads.googleads.client import GoogleAdsClient
# client = GoogleAdsClient.load_from_storage()
from google.ads.googleads.client import GoogleAdsClient
client = GoogleAdsClient.load_from_storage("credent.yaml")


import argparse
import sys
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
 
# Location IDs are listed here:
# https://developers.google.com/google-ads/api/reference/data/geotargets
# and they can also be retrieved using the GeoTargetConstantService as shown
# here: https://developers.google.com/google-ads/api/docs/targeting/location-targeting
_DEFAULT_LOCATION_IDS = ["2356"]  # location ID for India
# A language criterion ID. For example, specify 1000 for English. For more
# information on determining this value, see the below link:
# https://developers.google.com/google-ads/api/reference/data/codes-formats#expandable-7
_DEFAULT_LANGUAGE_ID = "1000"  # language ID for English
 
 





# [START generate_keyword_ideas]


customer_id = "4998884170"
language_id = _DEFAULT_LANGUAGE_ID
location_id = _DEFAULT_LOCATION_IDS
page_url = ''

def keywrodsGenerator(client, customer_id, location_ids, language_id, keyword_texts, page_url):

    keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")



    keyword_competition_level_enum = client.get_type(
        "KeywordPlanCompetitionLevelEnum"
    ).KeywordPlanCompetitionLevel
    keyword_plan_network = client.get_type(
        "KeywordPlanNetworkEnum"
    ).KeywordPlanNetwork.GOOGLE_SEARCH_AND_PARTNERS
    location_rns = _map_locations_ids_to_resource_names(client, location_ids)
    language_rn = client.get_service(
        "GoogleAdsService"
    ).language_constant_path(language_id)
     
    keyword_annotation = client.enums.KeywordPlanKeywordAnnotationEnum
     
    # Either keywords or a page_url are required to generate keyword ideas
    # so this raises an error if neither are provided.
    if not (keyword_texts or page_url):
        raise ValueError(
            "At least one of keywords or page URL is required, "
            "but neither was specified."
        )
     
     
     
    # Only one of the fields "url_seed", "keyword_seed", or
    # "keyword_and_url_seed" can be set on the request, depending on whether
    # keywords, a page_url or both were passed to this function.
    request = client.get_type("GenerateKeywordIdeasRequest")
    request.customer_id = customer_id
    request.language = language_rn
    request.geo_target_constants = location_rns
    request.include_adult_keywords = False
    request.keyword_plan_network = keyword_plan_network
    request.keyword_annotation = keyword_annotation
     
     
     
    # To generate keyword ideas with only a page_url and no keywords we need
    # to initialize a UrlSeed object with the page_url as the "url" field.
    if  not keyword_texts and page_url:
    	request.url_seed.url = page_url
 
    # To generate keyword ideas with only a list of keywords and no page_url
    # we need to initialize a KeywordSeed object and set the "keywords" field
    # to be a list of StringValue objects.
    if keyword_texts and not page_url:
        request.keyword_seed.keywords.extend(keyword_texts)
 
    # To generate keyword ideas using both a list of keywords and a page_url we
    # need to initialize a KeywordAndUrlSeed object, setting both the "url" and
    # "keywords" fields.
    if keyword_texts and page_url:
        # print(keyword_texts)
        # print(len(keyword_texts))
        request.keyword_and_url_seed.url = page_url
        request.keyword_and_url_seed.keywords.extend(keyword_texts)
 
    keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(
        request=request
    )
     
    list_keywords = []
    for idea in keyword_ideas:
        competition_value = idea.keyword_idea_metrics.competition.name
        list_keywords.append(idea)
     
    return list_keywords
 
def map_keywords_to_string_values(client, keyword_texts):
    keyword_protos = []
    for keyword in keyword_texts:
        string_val = client.get_type("StringValue")
        string_val.value = keyword
        keyword_protos.append(string_val)
    return keyword_protos
 
 
def _map_locations_ids_to_resource_names(client, location_ids):
    """Converts a list of location IDs to resource names.
    Args:
        client: an initialized GoogleAdsClient instance.
        location_ids: a list of location ID strings.
    Returns:
        a list of resource name strings using the given location IDs.
    """
    build_resource_name = client.get_service(
        "GeoTargetConstantService"
    ).geo_target_constant_path
    return [build_resource_name(location_id) for location_id in location_ids]
 
 
# if __name__ == "__main__":
#     # GoogleAdsClient will read the google-ads.yaml configuration file in the
#     # home directory if none is specified.
#     googleads_client = GoogleAdsClient.load_from_storage("credent.yaml")
 
#     parser = argparse.ArgumentParser(
#         description="Generates keyword ideas from a list of seed keywords."
#     )
 
#     # The following argument(s) should be provided to run the example.
#     parser.add_argument(
#         "-c",
#         "--customer_id",
#         type=str,
#         required=True,
#         help="The Google Ads customer ID.",
#     )
#     parser.add_argument(
#         "-k",
#         "--keyword_texts",
#         nargs="+",
#         type=str,
#         required=False,
#         default=[],
#         help="Space-delimited list of starter keywords",
#     )
#     # To determine the appropriate location IDs, see:
#     # https://developers.google.com/google-ads/api/reference/data/geotargets
#     parser.add_argument(
#         "-l",
#         "--location_ids",
#         nargs="+",
#         type=str,
#         required=False,
#         default=_DEFAULT_LOCATION_IDS,
#         help="Space-delimited list of location criteria IDs",
#     )
#     # To determine the appropriate language ID, see:
#     # https://developers.google.com/google-ads/api/reference/data/codes-formats#expandable-7
#     parser.add_argument(
#         "-i",
#         "--language_id",
#         type=str,
#         required=False,
#         default=_DEFAULT_LANGUAGE_ID,
#         help="The language criterion ID.",
#     )
#     # Optional: Specify a URL string related to your business to generate ideas.
#     parser.add_argument(
#         "-p",
#         "--page_url",
#         type=str,
#         required=False,
#         help="A URL string related to your business",
#     )
 
#     args = parser.parse_args()
 
#     try:
#         main(
#             googleads_client,
#             args.customer_id,
#             args.location_ids,
#             args.language_id,
#             args.keyword_texts,
#             args.page_url,
#         )
#     except GoogleAdsException as ex:
        # print(
#             f'Request with ID "{ex.request_id}" failed with status '
#             f'"{ex.error.code().name}" and includes the following errors:'
#         )
        # print(f'\tError with message "{error.message}".')
#         if error.location:
#             for field_path_element in error.location.field_path_elements:
                # print(f"\t\tOn field: {field_path_element.field_name}")
#         sys.exit(1)




# def matchWords(word1, word2):
# 	if(len(word1) != len(word2))






def getHead(word):
	l = len(word) -1
	s = ""
	for i in range(0,l):
		s = s + word[i]

	if(s == 'occasions' or s =="Occasions" or s =='Occasion' or s =='occasion'):
		return 0


	if(s == 'Relationships' or s =='Relationship' or s =='relationships' or s =='relationship'):

		return 1

	if(s == 'Material Type' or s =='Product Type' or s =='Material/Product Type' ):

		return 2

	if(s == 'Design/Style' or s =='Design' or s =='Style' or s =='Color' or s =='Pattern' or s =='Adjective' ):

		return 3
		
	if(s == 'Verb' or s =='Service'  ):

		return 4
		
	if(s == 'Place'  ):

		return 5

	if(s == 'Companies'  ):

		return 6
	
	return 7

		



def generateNGram(keyword_texts, k):
	print("Generating keywords for = ", keyword_texts)

	list_keywords = keywrodsGenerator(client, customer_id,location_id, language_id, keyword_texts, page_url)

	list_to_excel = []
	generated_keywords = []
	for x in range(len(list_keywords)):
		generated_keywords.append(list_keywords[x].text)

		# arr = [list_keywords[x].text, list_keywords[x].keyword_idea_metrics.avg_monthly_searches]
		# if(x < len(keyword_texts)):
			# arr.append(keyword_texts[x])


		# list_to_excel.append(arr)
	n = str(k)
	# pd.DataFrame(list_to_excel, columns = ["Keyword", "Average Searces","KeyPlanner"]).to_excel(f'keywords{n}.xlsx', header=True, index=False)

	d = {}

	exclude_words = {
		'for': 1,
		'and': 1,
		'nor': 1,
		'but': 1,
		'or': 1,
		'is': 1,
		'yet': 1,
		'so': 1,
		'are': 1, 
		'was': 1,
		'were': 1
	}

	for i in range(len(list_keywords)):
		word_list = list_keywords[i].text.split(" ")

		for word in word_list:
			if(d.__contains__(word)):
				d[word]+= list_keywords[i].keyword_idea_metrics.avg_monthly_searches
			else:
				d[word] = list_keywords[i].keyword_idea_metrics.avg_monthly_searches


	sorted_dict = dict(sorted(d.items(), key=lambda item: item[1]))
	sorted_keys = list(sorted_dict.keys())

	sorted_keys.reverse()




	list_to_excel = []
	ngram = []
	for x in sorted_keys:
		if x in exclude_words: 
			continue
		ngram.append([x, d[x]])
		# list_to_excel.append([x, d[x]])

	# pd.DataFrame(list_to_excel, columns = ["Keyword", "Average Searces"]).to_excel(f'nGram{n}.xlsx', header=True, index=False)
	# print()
	# print("generated keywords = ", generated_keywords)
	return generated_keywords, ngram


def throwAttributes(response, category):


	df = pd.read_excel(f'{category}.xlsx') # can also index sheet by name or fetch all sheets
	words = df['Keyword'].tolist()
	tag = df['tag'].tolist()

	head = 7
	
	for i in range(len(words)):
		head = 7
		if(tag[i] == 'm'):
			head = 1
		if(tag[i] == 'c'):
			head = 6
		if(tag[i] == 'a'):
			head  = 5
		if(tag[i] == 's'):
			head = 4
		if(tag[i] == 'd'):
			head = 3
		if(tag[i] == 'p'):
			head = 2
		response['message'][head].append(words[i])	





	# df = pd.read_excel('gptReply.xlsx') # can also index sheet by name or fetch all sheets
	# mylist = df['A'].tolist()
	# cleanedList = [x for x in mylist if str(x) != 'nan']
	# for i in range(0,len(cleanedList)):
	# 	cleanedList[i] = str(cleanedList[i])
	# head = 7
	# for word in cleanedList:
	# 	if(type(word) != bool and type(word)):
	# 		if(word[-1] == ':'):
	# 			head = getHead(word)
	# 		else:
	# 			response['message'][head].append(word)	

		


def filterKeywords(nWords, response):
	filtered_keywords  = []
	df = pd.read_excel('generated_keywords.xlsx') # can also index sheet by name or fetch all sheets
	mylist = df['Keyword'].tolist()
	keywords = [x for x in mylist if str(x) != 'nan']
	mylist = df['KeyPlanner'].tolist()
	keyPlanner = [x for x in mylist if str(x) != 'nan']




	negWords = {}
	for i in range(len(nWords)):
		negWords[nWords[i]] = 1

	for i in range(len(keyPlanner)):
		keyPlannerWords = keyPlanner[i].split(" ")
		adGroup = []
		keyPlannerWordsDict = {};
		for j in range(len(keywords)):
			for k in range(len(keyPlannerWords)):
				keyPlannerWordsDict[keyPlannerWords[k].lower()] = 0

			words = keywords[j].split(" ")
			k = 0
			flag = True 
			while(k < len(words) and flag):
				word = words[k]
				k+=1

				if(negWords.__contains__(word)):
					flag = False
				if(keyPlannerWordsDict.__contains__(word)):
					keyPlannerWordsDict[word] = 1

			if(flag):
				flag2 = True
				for x in keyPlannerWordsDict.keys():
					if(keyPlannerWordsDict[x] == 0):
						flag2 = False


				if(flag2):
					adGroup.append(keywords[j])
		filtered_keywords.append(adGroup)

	



def filter(nWords,nWordsDict, must_have, k,keywords, response, ad_group_words):

	if(must_have == ''):
		# print("must_have_word not provided")
		response['message'] = 'must_have not found'
		return

	filtered_keywords  = []
	print("filtering...")
	# keywords_file_name = 'keywords'+ str(k)+".xlsx"

	# df = pd.read_excel(keywords_file_name) # can also index sheet by name or fetch all sheets
	# mylist = df['Keyword'].tolist()
	# keywords = [x for x in mylist if str(x) != 'nan']
	# mylist = df['KeyPlanner'].tolist()
	# keyPlanner = [x for x in mylist if str(x) != 'nan']

	must_have_words = list(must_have.split(", "))
	# print("must_have_word = ",must_have_words)
	# print(keywords)

	skip = {}
	for nWord in nWords:
		for must in must_have_words:
			if(nWord in must):
				skip[nWord] = 1


	filtered_keywords = {}
	for keyword in keywords:

		preferred_keyword = True

		# print("keyword = ", keyword)

		for must in must_have_words:
			must = must.lower()
			if(keyword.find(must) == -1):
				# print("must not found = ", must)
				preferred_keyword = False

		for nWord in nWords:
			if(skip.__contains__(nWord)):
				# print("skipped = ", nWord)
				continue

			if(nWord in keyword):
				# print("nWrod found = ", nWord)
				preferred_keyword = False

		print("----                                     ----")

		if(preferred_keyword):
			done = False
			for ad_group_word in ad_group_words:
				x = list(map(str, keyword.split(" ")))
				for word in x:
					if(ad_group_word == word):
						done = True
						if(not filtered_keywords.__contains__(ad_group_word)): 
							filtered_keywords[ad_group_word] = [keyword]
						else:
							filtered_keywords[ad_group_word].append(keyword)
			if(done == False):
				if(not filtered_keywords.__contains__(must_have_words[0])):
					filtered_keywords[must_have_words[0]] = [keyword]
				else:
					filtered_keywords[must_have_words[0]].append(keyword)

	print(filtered_keywords)

	return (filtered_keywords)


	






all_generated_keywords = []
all_nGrams = []

app = Flask(__name__)
cors = CORS(app)
@app.route('/form', methods=['GET', 'POST'])
def form():
	response = {
		"message": [[],[],[],[],[],[],[],[],[],[],[]]
	}
	
	my_dict = json.loads(request.data.decode('utf-8'))
	name=    (my_dict['name'])
	allUsp = (my_dict['allUsp'])
	print(allUsp)
	productKeywords = (my_dict['productKeywords'])

	category = (my_dict['category'])
	
	keyword_texts = list(productKeywords.split(", "))
	usp_keywords = {}
	all_keywords = [keyword_texts]
	nWords = list(my_dict['price'].keys());
	nG = my_dict['price'];
	nWords = []
	ad_group_words = []
	for x in nG.keys():
		if(x == 'Both' or x == 'Offline' or x == 'Online'):
			continue
		if(nG[x] == 1):
			nWords.append(x)
		elif(nG[x] == 2):
			ad_group_words.append(x)



	print(nWords)
	print(ad_group_words)

	customer_id = "4998884170"
	language_id = _DEFAULT_LANGUAGE_ID
	location_id = _DEFAULT_LOCATION_IDS

	for x in allUsp:
		
		usp_keywords[allUsp[x]['identifierWords']] = list(allUsp[x]['uspKeyWords'].split(", "))
		all_keywords.append(usp_keywords[allUsp[x]['identifierWords']])
		
	temp = []
	for listOfKeywords in all_keywords:
		if(len(listOfKeywords) == 1 and listOfKeywords[0] == ''):
			continue
		temp.append(listOfKeywords)

	all_keywords = temp

	brand_level_keywords = []
	for listOfKeywords in all_keywords:
		for keyword in listOfKeywords:
			brand_level_keywords.append(keyword)



	if(len(brand_level_keywords) != 0):
		all_keywords.append(brand_level_keywords)
	
	# print("all_keywords=", all_keywords)


	if(len(nWords) == 0):
		k = 0
		for keyword_texts in all_keywords:
			if(k == len(all_keywords )-1):
				continue
			generated_keywords, nGram = generateNGram(keyword_texts, k)
			k+=1	
			all_generated_keywords.append(generated_keywords)
			all_nGrams.append(nGram)

		response['message'][8] = all_generated_keywords
		response['message'][9] = all_nGrams

		throwAttributes(response, category)



	# if(len(nWords) == 0):
	# 	k = 0
	# 	if(len(all_keywords) != 0):
	# 		for keyword_texts in all_keywords:

	# 			generateNGram(keyword_texts, k)
	# 			k+=1
	# 	else:	
	# 		throwAttributes(response, category)


	else:
		negWords = {}
		for i in range(len(nWords)):
			negWords[nWords[i]] = 1


		all_filtered_keywrods = []
		k = 0
		must_have = name 
		keywords_list = list(map(str, productKeywords.split(", ")))
		print("all_generated_keywords = ", all_generated_keywords)
		print(k)
		print("must_have = ", must_have)
		print("all_generated_keywords = ", all_generated_keywords[k])

		filtered_keywords = filter(nWords,negWords, must_have, k,all_generated_keywords[k],response, ad_group_words)	
		k+=1
		all_filtered_keywrods.append(filtered_keywords)
		for x in usp_keywords.keys():

			must_have = x
			keywords_list = usp_keywords[x]
			print(k)
			print("must_have = ", must_have)
			print("all_generated_keywords = ", all_generated_keywords[k])

			filtered_keywords = filter(nWords,negWords, must_have, k, all_generated_keywords[k], response, ad_group_words)
			all_filtered_keywrods.append(filtered_keywords)
			k+=1
		response['message'][9] = nWords
		response['message'][10] = all_filtered_keywrods


	return jsonify(response), 200
	


	
	# if(len(nWords) != 0):
		# filterKeywords(nWords)













# import openai  
# openai.api_key = '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'
# messages = [ {"role": "system", "content":  
#               "You are a intelligent assistant."} ] 
# while True: 
#     message = input("User : ") 
#     if message: 
#         messages.append( 
#             {"role": "user", "content": message}, 
#         ) 
#         chat = openai.ChatCompletion.create( 
#             model="gpt-3.5-turbo", messages=messages 
#         ) 
#     reply = chat.choices[0].message.content 
    # print(f"ChatGPT: {reply}") 
#     messages.append({"role": "assistant", "content": reply}) 

app.run(host= '0.0.0.0',port = 5000)
