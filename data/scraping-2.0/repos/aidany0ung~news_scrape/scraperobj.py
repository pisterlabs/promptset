# Create Scraper class
from urllib.request import Request, urlopen
import datetime
import requests
import json
import pymongo
from pymongo import MongoClient
from bs4 import BeautifulSoup
import openai
import os
import flair
import random
import re

def makeLocationRequest(search_term):
    if search_term == None:
        return None
    if search_term == "":
        return None
    if search_term == " ":
        return None
    formatted = search_term.replace(" ","%20")
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input="+formatted+"&inputtype=textquery&fields=geometry&key=AIzaSyAp5g9TKFj_nBJV8-bh1EMiCv79J7mMbC0"

    payload={}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)

    geo = json.loads(response.text)['candidates']
    if len(geo) == 0:
        return None
    geo = geo[0]['geometry']['location']
    [lat,long] = [geo['lat'],geo['lng']]
    return [lat,long]

# Load locations and nations.json
with open('backend/locations.json') as f:
    locations = json.load(f)

with open('backend/nations.json') as f:
    nations = json.load(f)


class Article:
    def __init__(self,title,author,date,href,source,img= None):
        self.title = title
        self.author = author
        self.date = date
        self.img = img
        self.href = href
        self.source = source
        self.lat = None
        self.long = None

    def iteration(self,word):
        word = word.replace(",","")
        word = word.replace(".","")
        word = word.replace('"',"")
        word = word.replace("'s","")
        word = word.replace("'","")
        word = word.replace("’s","")
        word = word.replace("’","")
        word = word.replace("“","")
        word = word.replace("”","")
        word = word.replace("(","")
        word = word.replace(")","")
        word = word.replace("[","")
        word = word.replace("]","")
        word = word.replace("{","")
        word = word.replace("}","")
        word = word.replace(":","")
        word = word.replace(";","")
        word = word.replace("!","")
        word = word.replace("?","")
        word = word.replace("—","")
        word = word.replace("-","")
        word = word.replace("–","")
        word = word.replace("…","")
        word = word.replace("‘","")
        word = word.replace("’","")

        if word in nations:
            country = nations[word]
            if len(locations[country])>0:
                loc_found = True
                self.lat = locations[country][0]
                self.long = locations[country][1]
                return [True, False]
            else:
                result = makeLocationRequest(country)
                if result != None:
                    loc_found = True
                    loc_update = True
                    self.lat = result[0]
                    self.long = result[1]
                    locations[country] = [self.lat,self.long]
                    return [True, True]
        if word in locations:
            if len(locations[word])>0:
                loc_found = True
                self.lat = locations[word][0]
                self.long = locations[word][1]
                return [True, False]
            else:
                result = makeLocationRequest(word)
                if result != None:
                    loc_found = True
                    loc_update = True
                    self.lat = result[0]
                    self.long = result[1]
                    locations[word] = [self.lat,self.long]
                    return [True, True]
        seg = self.title.split(" ")
        for i in range(len(seg)-1):
            double = seg[i]+" "+seg[i+1]
            if double in nations:
                country = nations[double]
                if len(locations[country])>0:
                    loc_found = True
                    self.lat = locations[country][0]
                    self.long = locations[country][1]
                    return [True, False]
                else:
                    result = makeLocationRequest(country)
                    if result != None:
                        loc_found = True
                        loc_update = True
                        self.lat = result[0]
                        self.long = result[1]
                        locations[country] = [self.lat,self.long]
                        return [True, True]
            if double in locations:
                if len(locations[double])>0:
                    loc_found = True
                    self.lat = locations[double][0]
                    self.long = locations[double][1]
                    return [True, False]
                else:
                    result = makeLocationRequest(double)
                    if result != None:
                        loc_found = True
                        loc_update = True
                        self.lat = result[0]
                        self.long = result[1]
                        locations[double] = [self.lat,self.long]
                        return [True, True]
        return [False, False]

    def attachLatLong(self):
        loc_found = False
        loc_update = False
        self.lat = None
        self.long = None

        for word in self.title.split(" "):
            [loc_found, temp] = self.iteration(word)
            if temp:
                loc_update = True
            if loc_found:
                break
        if not loc_found:
            result = makeLocationRequest(self.author)
            if result != None:
                loc_found = True
                self.lat = result[0]
                self.long = result[1]
                return [True, False]
        if loc_update == True:
            with open('backend/locations.json', 'w') as f:
                json.dump(locations, f)
        
        
        
    # Write a method to get the latitude and longitude of the article
    def sentimentAnalysis(self,inferencetype):
        if inferencetype=='random':
            random.randint(0,2)-1
            self.sentiment = random.randint(0,2)-1
        elif inferencetype=='openai':
            print("Trying to get lat/long from AI for '" + str(self.title)+"'")
            openai.api_key = os.getenv("OPENAI_API_KEY")
            try:
                usage = " ".join([str(self.title), "by", str(self.author), "->"])
                result = openai.Completion.create(model = "ada:ft-personal:sentibot-2023-07-07-20-34-14",prompt = usage,stop = ["###"])
                result = result['choices'][0]['text']
                result = result.strip()
                result_data = 0
                if result == "positive":
                    result_data = 1
                elif result == "negative":
                    result_data = -1
                self.sentiment = result_data
            except Exception as e:
                print(e)
                self.sentiment = 0
                print("Error with AI")
        elif inferencetype=='flairai':
            flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
            s = flair.data.Sentence(self.title)
            flair_sentiment.predict(s)
            total_sentiment = str(s.labels).split("/")[1]
            # Isolate the sentiment
            total_sentiment = total_sentiment[total_sentiment.find("'")+1:total_sentiment.find("' ")]
            if total_sentiment == "POSITIVE":
                self.sentiment = 1
            elif total_sentiment == "NEGATIVE":
                self.sentiment = -1
            else:
                self.sentiment = 0
        sentiment = 0
    # Override the equals method based on title and author
    def __eq__(self, other):
        if not isinstance(other, Article):
            return False
        return self.title == other.title and self.author == other.author
    
    # Override the str method
    def __str__(self):
        if self.author == None:
            return self.title + " on " + self.date + " from " + self.source
        return self.title + " by " + self.author + " on " + self.date + " from " + self.source + " (url: " + self.href + ")"
    
    # Override the hash method
    def __hash__(self):
        return hash((self.title, self.author))
    
    def toDictionary(self):
        return {"title": self.title, "author": self.author, "date": self.date, "img": self.img, "href": self.href, "source": self.source, "coordinates": [self.long, self.lat], "sentiment": self.sentiment}

class Scraper:
    def __init__(self, date):
        self.date = date
        self.storage = set()
        self.datasources = {"EcoWatch": {"homeURL": "https://www.ecowatch.com", "findBy": "", "titleClass": "", "authorClass":"","imgClass":"","dateClass":""},
                            "ENN": {"homeURL": "https://www.enn.com", "findBy": ["/articles/",""], "titleClass": "", "authorClass":"","imgClass":"","dateClass":""},
                            "Grist": {"homeURL": "https://www.grist.org", "findBy": "", "titleClass": "", "authorClass":"","imgClass":"","dateClass":""},
                            "MongoBay": {"homeURL": "https://news.mongabay.com", "findBy": "", "titleClass": "", "authorClass":"","imgClass":"","dateClass":""},
        }
        self.functions = {"ENN": self.scrapeENN,
                          "InsideClimateNews": self.scrapeICN,
                          "Grist": self.scrapeGrist,
                          "MongaBay": self.scrapeMongoBay,}

    # Convert date in format %d %B %Y to MM/DD/YYYY
    def convertDate(self, date):
        # Convert date to datetime object
        date = datetime.datetime.strptime(date, "%d %B %Y")
        # Convert date to string in format MM/DD/YYYY
        date = date.strftime("%m/%d/%Y")
        return date

    # Scrape
    def run(self,src = []):
        if src == []:
            src = self.datasources.keys()
        for source in src:
            self.functions[source]()
    
    def scrapeGrist(self):
        source = "Grist"
        # Get all links from datasources['homeURL'] using BeautifulSoup
        url = self.datasources[source]['homeURL']
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        soup = BeautifulSoup(webpage, 'html.parser')
        links = soup.find_all('a', href=True)  # Find all anchor tags with 'href' attribute
        links = [link['href'] for link in links]
        
        # Find all links, get their raw hrefs
        linky = set()
        for link in links:
                if "author" not in link and "#" not in link and "about" not in link and "fiction" not in link and "fix" not in link:
                    if "https://grist.org/" in link:
                        if link.count("/") > 4:
                            linky.add(link)
        hrefs = list(linky)
        print(hrefs)
        # For each href, get the title, author, date, and image
        for href in hrefs:
            try:
                page = requests.get(href)
            except:
                print("Error with href: "+href)
            
            articleSoup = BeautifulSoup(page.content, "html.parser")
            try:
                author = articleSoup.find("span", class_="contributor-info__name").get_text()
                author = author.strip()
            except:
                author = None
                # Date is the time element within the dd tag with class name: published
            try:
                date = articleSoup.find("dd", class_="article-meta__item-value").get_text()
                date = date.strip()
                date = datetime.datetime.strptime(date, "%b %d, %Y")
                # Convert date to string in format MM/DD/YYYY
                date = date.strftime("%m/%d/%Y")
            except:
                date = None
                # Author is the a tag within the dd tag with class name: author
            title = articleSoup.find("title").get_text().split(" | ")[0]
                # Write this for set
                # strip newline and spaces from the sides of all the strings
            title = title.strip()
            
                #date = self.convertDate(date)
            try:
                img = articleSoup.find("figure", class_="topper-featured-image__figure").find("img").get("src")
                img = img.strip()
            except:
                img = None
            
        #except:
        #    print("Error with page parsing: "+href)
        #    continue
            # Check to see if date is today
            #if date == "1 July 2023": #self.date.strftime("%d %B %Y"):
            if title != None and date != None:
                self.storage.add(Article(title,author,date,href,source,img=img))
    
    def scrapeENN(self):
        source = "ENN"
        # Get all links from datasources['homeURL'] using BeautifulSoup
        URL = self.datasources[source]['homeURL']
        page = requests.get(URL)
        homeSoup = BeautifulSoup(page.content, "html.parser")
        
        # Find all links, get their raw hrefs
        links = homeSoup.find_all("a")
        links = [link.get('href') for link in links]
        hrefs = []
        for link in links:
            if isinstance(link, str):
                if link[:len(self.datasources[source]['findBy'][0])] == self.datasources[source]['findBy'][0]:
                    hrefs.append("https://www.enn.com"+link)
        hrefs = list(set(hrefs))

        # For each href, get the title, author, date, and image
        for href in hrefs:
            try:
                page = requests.get(href)
            except:
                print("Error with href: "+href)
            
            articleSoup = BeautifulSoup(page.content, "html.parser")
            try:
                title = articleSoup.find("h1", class_="article-title").get_text()
                # Date is the time element within the dd tag with class name: published
                date = articleSoup.find("dd", class_="published").find("time").get_text()
                # Author is the a tag within the dd tag with class name: author
                author = articleSoup.find("dd", class_="createdby").find("span").get_text()
                # Write this for set
                img = "https:"+articleSoup.find(class_="item-image").find("img").get("src")
                # strip newline and spaces from the sides of all the strings
                title = title.strip()
                date = date.strip()
                date = self.convertDate(date)
                author = author.strip()
            except:
                print("Error with page parsing: "+href)
                continue
            # Check to see if date is today
            #if date == "1 July 2023": #self.date.strftime("%d %B %Y"):
            self.storage.add(Article(title,author,date,href,source,img=img))

    
    def scrapeMongoBay(self):
        source = "MongoBay"
        url = self.datasources[source]['homeURL']
        # Get all links from datasources['homeURL'] using BeautifulSoup
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        soup = BeautifulSoup(webpage, 'html.parser')
        links = soup.find_all('a', href=True)  # Find all anchor tags with 'href' attribute
        links= [link['href'] for link in links]
        regex_pattern = r'https:\/\/news\.mongabay\.com\/[0-9]{4}\/[0-9]{2}\/'
        links = [link for link in links if re.match(regex_pattern, link)]
        temp = []
        for link in links:
            if isinstance(link, str):
                if link.removeprefix("https://news.mongabay.com")[1:5] == datetime.datetime.now().strftime("%Y") and link.removeprefix("https://news.mongabay.com")[6:8] == datetime.datetime.now().strftime("%m"):
                    temp.append(link)
        hrefs = list(set(temp))

        print(hrefs)
        # For each href, get the title, author, date, and image
        for href in hrefs:
            try:
                req = Request(href, headers={'User-Agent': 'Mozilla/5.0'})
                webpage = urlopen(req).read()
            except:
                print("Error with href: "+href)

            articleSoup = BeautifulSoup(webpage, 'html.parser')

            #try:
            title = articleSoup.find("div", class_="article-headline").find("h1").get_text()
            # Date is the time element within the dd tag with class name: published
            meta = articleSoup.find("div", class_="single-article-meta").get_text().split(" on ")
            date = meta[-1].split("|")[0]
            # Author is the a tag within the dd tag with class name: author
            author = meta[0].split(" by ")[0].removeprefix('\nby ')
            # Write this for set
            # strip newline and spaces from the sides of all the strings
            title = title.strip()
            date = date.strip()
            if len(date.split(" ")[0]) == 1:
                date = "0"+date
            date = datetime.datetime.strptime(date, "%d %B %Y")
            # Convert date to string in format MM/DD/YYYY
            date = date.strftime("%m/%d/%Y")
            author = author.strip()
            temp = articleSoup.find("div", class_="article-cover-image")
                # Get the first div within temp
            temp = temp.find("div")
                # Get the style of the div as a string
            temp = temp.get("style")
                # Get the url of the image
            img = temp[temp.find("url(")+5:temp.find(")")-1]
            print(temp, img)
            #except:
                #print("Error with page parsing: "+href)
                #continue
            # Check to see if date is today
            #if date == "1 July 2023": #self.date.strftime("%d %B %Y"):
            self.storage.add(Article(title,author,date,href,source,img=img))

    def scrapeICN(self):
        source = 'Inside Climate News'
        URLs = ["https://insideclimatenews.org/category/science/","https://insideclimatenews.org/category/fossil-fuels/","https://insideclimatenews.org/category/justice/","https://insideclimatenews.org/category/politics-policy/","https://insideclimatenews.org/category/clean-energy/"]
        hrefs = set()
        for url in URLs:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            webpage = urlopen(req).read()
            soup = BeautifulSoup(webpage, 'html.parser')
            links = soup.find_all('a', href=True)  # Find all anchor tags with 'href' attribute
            links= [link['href'] for link in links]
            for i in links:
                # if i starts with "https://insideclimatenews.org/news/")
                if i.startswith("https://insideclimatenews.org/news/"):
                    t = i.removeprefix("https://insideclimatenews.org/news/")
                    ## If the first 8 characters of t are numbers:
                    if t[:8].isdigit():
                        # Convert the first 8 characters of t to a datetime object with format DDMMYYYY
                        date = datetime.datetime.strptime(t[:8], '%d%m%Y')
                        # Check if the date is within the last 14 days
                        if date > datetime.datetime.today() - datetime.timedelta(days=14):
                            hrefs.add(i)
        print(hrefs)
        # For each href, get the title, author, date, and image
        for href in hrefs:
            try:
                page = requests.get(href)
            except:
                print("Error with href: "+href)
            
            articleSoup = BeautifulSoup(page.content, "html.parser")
            try:
                title = articleSoup.find("h1", class_="entry-title").get_text()
                # Date is the time element within the dd tag with class name: published
                date = articleSoup.find("div", class_="date").find("time").get_text()
                # Author is the a tag within the dd tag with class name: author
                try:
                    author = articleSoup.find("div", class_="byline").find("a").get_text()
                    author = author.strip()
                except:
                    author = None
                # Write this for set
                tempo = articleSoup.find("figure",class_="entry-featured-image")
                temp = tempo.findAll("img")[-1]
                img = temp.get("src")
                # strip newline and spaces from the sides of all the strings
                title = title.strip()
                date = date.strip()
                # Convert July 24, 2023 to 07/24/2023
                date = datetime.datetime.strptime(date, "%B %d, %Y")
                # Convert date to string in format MM/DD/YYYY
                date = date.strftime("%m/%d/%Y")
            except:
                print("Error with page parsing: "+href)
                continue
            # Check to see if date is today
            #if date == "1 July 2023": #self.date.strftime("%d %B %Y"):
            self.storage.add(Article(title,author,date,href,source,img=img))

        
    def getArticles(self):
        for article in self.storage:
            print(article)
    
    def upload(self, connection,inferencetype = 'random'):

        # Print date in MM/DD/YYYY format
        date = self.date.strftime("%m/%d/%Y")
        
        # Create database
        db = connection["news"]

        # Insert articles
        for article in self.storage:
            if article.img == None:
                continue
            collection = db[article.date]
            # Check to see if the article is already in the database based on title and author
            if collection.find_one({"title": article.title, "author": article.author}) == None:
                article.sentimentAnalysis(inferencetype)
                article.attachLatLong()
                collection.insert_one(article.toDictionary())
        