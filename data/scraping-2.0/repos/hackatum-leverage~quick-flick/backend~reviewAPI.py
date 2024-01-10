
import re
import json
import openai
import mongo
import requests
from bs4 import BeautifulSoup
import urllib.request, json

mdb_url = "https://api.themoviedb.org/3/"
mdb_key = "5df139106aa0fb2f1b015f82b6bf0a7a"

CLEANR = re.compile('<.*?>')
NUMREVIEWS = 4

def get_imdb_id_movie(id):
    with urllib.request.urlopen(mdb_url+ "movie/" + str(id) + "?api_key=" + mdb_key) as url:
        req = json.loads(url.read().decode())
        ret = req["imdb_id"]
    return ret

def get_imdb_id_tv(id):
    with urllib.request.urlopen(mdb_url+ "tv/" + str(id) + "/external_ids" + "?api_key=" + mdb_key) as url:
        req = json.loads(url.read().decode())
        ret = req["imdb_id"]
    return ret

def cleanHtml(_rawHtml):
        cleanText = re.sub(CLEANR, '', _rawHtml)
        return cleanText

def getReviewData(_tmdbID):
    try:
        entityName, entityVersion = mongo.check_tmdb_name(_tmdbID)
    except Exception as e:
        return []

    imdbID = 0
    if entityVersion == 0:
        # Movie
        try:
            imdbID =  get_imdb_id_movie(_tmdbID)
        except Exception as e:
            return []
    else:
        # Series
        try:
            imdbID = get_imdb_id_tv(_tmdbID)
        except Exception as e:
            return []

    URL = "https://www.imdb.com/title/" + imdbID + "/reviews?ref_=tt_sa_3"

    response = requests.get(URL)
    if response.status_code != 200:
        return []

    try:
        page = requests.get(URL)
    except Exception as e:
        return []

    # Get necessary data of webpage
    soup = BeautifulSoup(page.content, "html.parser")
    # Get overall review box
    reviewsBox = soup.find(class_="lister-list")
    # Get Usernames
    userNames = reviewsBox.find_all("span", class_="display-name-link")
    # Get corresponding reviews
    reviews = reviewsBox.find_all("div", class_="text show-more__control")

    # Max number of elements
    if len(reviews) > NUMREVIEWS:
        reviews = reviews[0:NUMREVIEWS]
        userNames = userNames[0:NUMREVIEWS]

    # Convert to string list
    reviewList = []
    for review in reviews:
        reviewList.append(cleanHtml(str(review)))
    userNameList = []
    for userName in userNames:
        userNameList.append(cleanHtml(str(userName)))

    # Settings for openai
    openai.api_key = "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"

    # Get data from open ai
    responseList = []
    userNameCounter = 0
    counter = 0
    for review in reviewList:
        response = openai.Completion.create(
            model = "text-davinci-002",
            prompt = "Briefly summarize the positive aspects of the following review in first person: " + review,
            temperature = 0.7,
            max_tokens = 26,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0
        )

        # Try to get data out of response
        try:
            reviewString = response['choices'][0]['text'].replace("\n", "")
            reviewString = reviewString.replace("\n", "")
            reviewStringList = reviewString.rsplit('.', 1)

            if len(reviewStringList) == 0:
                reviewString = "Just a marvellous movie. Nothing more to say!"
            elif len(reviewStringList) == 1:
                reviewString = "".join(reviewStringList) + "."
            else:
                reviewString = "".join(reviewStringList[:-1]) + "."

            responseJson = {
                "name": userNameList[userNameCounter],
                "review": reviewString,
            }
            responseList.append(responseJson)
            userNameCounter = userNameCounter + 1
            if counter >= 4:
                break
        except Exception as e:
            print("There was an error while summarizing data from openai")

    return responseList

def getMovieDescription(_tmdbID, _mode):
    # _mode = 0 -> short text
    # _mode = 1 -> adjectives

    try:
        entityName, entityVersion = mongo.check_tmdb_name(_tmdbID)
    except Exception as e:
        return []

    nameList = ['movie', 'series']

    if entityName is None:
        return ["-1"]

    # Settings for openai
    openai.api_key = "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"

    prompt = ""
    if _mode == 0:
        prompt = "name three reasons why people love " + entityName + " as a " + nameList[entityVersion] + ":"
    else:
        prompt = "name three adjectives why people love " + entityName + " as a " + nameList[entityVersion] + ":"

    response = openai.Completion.create(
            model = "text-davinci-002",
            prompt = prompt,
            temperature = 0.5,
            max_tokens = 48,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0
        )

    try:
        processedResponse = response['choices'][0]['text'].replace("\n", "")

        if "1. " not in processedResponse and "2. " not in processedResponse and "3. " not in processedResponse:
            return ["-1"]
        
        processedResponse = processedResponse.replace("1", "")
        processedResponse = processedResponse.replace("2", "")
        processedResponse = processedResponse.replace("3", "")
        processedResponse = processedResponse.split(". ")

        # Delete unnecessary elements
        for element in processedResponse:
            if not element:
                processedResponse.remove(element)

        # clean data one more time
        for idx, element in enumerate(processedResponse):
            processedResponse[idx] = processedResponse[idx].replace(".", "")

        return processedResponse
    except Exception as e:
        return []


#with open('openAIReponse.json', 'w') as f:
#        json.dump(getReviewData(0), f)

if __name__ == "__main__":
    print("Brrrrumm")
    #print(getMovieDescription(157336, 1))
    #print(getReviewData(157336))