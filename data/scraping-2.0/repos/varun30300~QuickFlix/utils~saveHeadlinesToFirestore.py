import os 
import requests
import firebase_admin
from firebase_admin import credentials, firestore
import hashlib
import openai


KEY = os.getenv('NEWS_API_KEY')
openai.api_key = os.getenv("NewsSummarization")

cred = credentials.Certificate("quickflick-19fdd-11e52bcd6a8b.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# reading the first page
curr_page = "https://newsdata.io/api/1/news?apikey="+ KEY +"&country=ca&language=en"

page_no = 1
while page_no <= 5 : 
    response = requests.get(curr_page).json()

    for article in response['results']:
        title = article['title']
        link = article['link']
        keywords = article['keywords']
        author = article['creator']
        description = article['description']
        content = article['content']
        pubDate = article['pubDate'].split(' ')[0]   # yyyy-mm-dd
        image_url = article['image_url']
        source_id = article['source_id']
        category = article['category']
        country = article['country']
        language = article['language']

        if content is not None:
            gptResponse = openai.Completion.create(
                model="text-davinci-003",
                prompt=content+"\n\nTl;dr",
                temperature=1,
                max_tokens=200,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=1
            )
            print("BAMMM, ur life summarized")

        else :
            print("Le content ceat null")
        
        summarization = gptResponse["choices"][0]["text"]

        
        # hashing the title to generate a unique id
        m = hashlib.md5()
        m.update(title.encode('utf-8'))
        docID = str(m.hexdigest())
        

        # print(docID)
        data = {
            "docId" : docID,
            "title" : title,
            "link" : link ,
            "keywords" : keywords ,
            "author" : author, 
            "description" : description, 
            "content" : content, 
            "pubDate" : pubDate, 
            "image_url" : image_url, 
            "source_id" : source_id, 
            "category" : category, 
            "country" : country, 
            "language" : language,
            "summarization" : summarization
        }


        print("title: ",docID)
        print(pubDate)
        collection = db.collection(pubDate)
        
        if len(collection.get()) == 0:
            latest = db.collection("latest").document("solo").set({"date":pubDate})
        # check for get, if length of get is 0, this is the latest news date


        doc = collection.document(docID)
        

    # if in pubDate collection if there is a doc with id link, ignore
        if (doc.get().exists):
            print("Old article, doing nothing")
        else :
            print("New article, adding it to the database")
            doc.set(data)
            print("added")


    
    # else in pubdate collection add the link doc 
    # break
    page_no += 1
    if  response['nextPage'] : 
        curr_page = "https://newsdata.io/api/1/news?apikey="+ KEY +"&country=ca&language=en&page=" + response['nextPage']
    else :
        break


# iterate and check if link for that date already exists, if yes ignore , if no add that to the news 
#test