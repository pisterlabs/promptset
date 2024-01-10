import pandas as pd
import openai
import os
from selenium import webdriver
from bs4 import BeautifulSoup
from gensim.summarization import summarize

df = pd.read_csv("ingredients.csv")
ingredients = df.values

PATH = "chromedriver.exe"
driver = webdriver.Chrome(PATH)
classify_prompt = os.environ["CLASSIFY_PROMPT"]
openai.api_key = os.environ["OPENAI_KEY"]
environment_prompt = os.environ["ENVIRONMENT_PROMPT"]
nutrition_prompt = os.environ["NUTRITION_PROMPT"]

for ingredient in ingredients: 
    # all the fields that will be collected for each ingredient
    info = {
        "name": [],
        "categories": [],
        "environment_titles": [],
        "environment_summaries": [],
        "environment_urls": [],
        "environment_sentiment": [],
        "nutrition_titles": [],
        "nutrition_summaries": [],
        "nutrition_urls": [],
        "nutrition_sentiment": [],
    }

    # classify each ingredient into various categories 
    name = ingredient[0]
    new_prompt = classify_prompt + "\n" + name + ":"
    info["name"].append(name)

    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt= new_prompt,
            temperature=0,
            max_tokens=15,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        temp = response.choices[0].text.split(",")
        for i in temp: 
            info["categories"].append(i.strip())
    except:
        info["categories"] = []

    # getting the environmental impact data based on top search results from the web
    # webscraping is done using selenium webdriver
    base_url = "https://duckduckgo.com/html?q="
    environment_url = base_url + "environmental+impact+of+" + name.replace(" ", "+")
    driver.get(environment_url)
    results = driver.find_elements_by_class_name("result__url")

    for i in range(0, 3):
        info["environment_urls"].append(results[i].text)

    # content of the websites from top search results are taken to be analyzed
    for url in info["environment_urls"]:
        temp = "https://" + url
        driver.get(temp)
        info["environment_titles"].append(driver.title)

        # BeautifulSoup is used to extract all the relevant information from the website
        page = driver.page_source
        soup = BeautifulSoup(page, features="lxml")
        tags = soup.find_all("p")   
        text = [tag.get_text().strip() for tag in tags]
        
        # Any sentences that contain these words will be removed
        remove = ["replies","share","advertising", "cookies", "privacy","updates", "analytics", "third-party", "feedback", "email", "e-mail", "facebook", "youtube", "twitter","unsubscribe","parties"]

        sentence_list = [sentence for sentence in text if not "\n" in sentence]
        sentence_list = [sentence for sentence in text if "." in sentence]
        
        for i in range(len(sentence_list)):
            words = sentence_list[i].split(" ")
            for word in words:
                temp = word.lower().strip().strip(",.:/")
                if temp in remove:
                    sentence_list[i] = ""

        # OpenAI's API is used to analyze the environmental sentiment for each ingredient's articles
        try: 
            page = " ".join(sentence_list)
            summary = summarize(page, ratio=0.3)
            text = environment_prompt + summary + "\nRATING:"

            response = openai.Completion.create(
                engine="davinci",
                prompt=text,
                temperature=0.3,
                max_tokens=1,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            info["environment_sentiment"].append(response.choices[0].text)

        except:
            print("ERROR")
            summary = "Could Not Retrieve Summary."
            info["environment_sentiment"].append("ERROR")

        info["environment_summaries"].append(summary)
        print("----------------------------------------------------------------------------------")

    # getting the nutritional benefits information based on top search results from the web
    # webscraping is done using selenium webdriver
    base_url = "https://duckduckgo.com/html?q="
    environment_url = base_url + "nutritional+benefits+of+" + name.replace(" ", "+")
    driver.get(environment_url)
    results = driver.find_elements_by_class_name("result__url")
    results = driver.find_elements_by_class_name("result__url")
    
    for i in range(0, 3):
        info["nutrition_urls"].append(results[i].text)

    # content of the websites from top search results are taken to be analyzed
    for url in info["nutrition_urls"]:
        temp = "https://" + url
        driver.get(temp)
        info["nutrition_titles"].append(driver.title)

        # BeautifulSoup is used to extract all the relevant information from the website
        page = driver.page_source
        soup = BeautifulSoup(page, features="lxml")
        tags = soup.find_all("p")   
        text = [tag.get_text().strip() for tag in tags]
        
        # Any sentences that contain these words will be removed
        remove = ["replies","share","advertising", "cookies", "privacy","updates", "analytics", "traffic", "thank", "third-party", "feedback", "email", "e-mail", "facebook", "youtube", "twitter","unsubscribe","parties"]
        sentence_list = [sentence for sentence in text if "." in sentence]

        for i in range(len(sentence_list)):
            words = sentence_list[i].split(" ")
            for word in words:
                temp = word.lower().strip().strip(",.:/")
                if temp in remove:
                    sentence_list[i] = ""
        page = " ".join(sentence_list)
        summary = summarize(page, ratio=0.3)

        if not summary: 
            print("ERROR")
            summary = "Could Not Retrieve Summary."
            info["nutrition_sentiment"].append("ERROR")
        else:
             # OpenAI's API is used to analyze the environmental sentiment for each ingredient's articles
            try:
                text = nutrition_prompt + summary + "\nRATING:"
                response = openai.Completion.create(
                    engine="davinci",
                    prompt=text,
                    temperature=0.3,
                    max_tokens=1,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
        
                info["nutrition_sentiment"].append(response.choices[0].text)
            except:
                info["nutrition_sentiment"].append("ERROR")

        info["nutrition_summaries"].append(summary)
        
        print("-------------------------------------------------------------")
    
    # the information and analysis for each ingredient is written into its own json file using pandas
    dp = pd.DataFrame.from_dict(info, orient="index")
    dp = dp.transpose()
    temp = dp.to_json(info["name"][0] + ".json")

    # master["ingredients"].append(json)
    # dp1 = pd.DataFrame.from_dict(master, orient="index")
    # dp1 = dp.transpose()

    # temp1 = dp.to_json("ingredient_data.json")

