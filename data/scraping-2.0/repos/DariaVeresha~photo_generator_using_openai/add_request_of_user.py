from pymongo.mongo_client import MongoClient
import random
import datetime
from app_generator import document, PROMPT
from enter_to_mongoDB import url
#This database already has some images that I imported into it using this code!!!
#create connection to MongoDB-online db(From any divice)

try:
    uri = url
    # Create a new client and connect to the server
    client = MongoClient(uri)
    # create db and collection
    db = client.db_photosGener
    collection = db.photos
    if collection.count_documents({}) == 1000:
        for document in collection.find({}):
            print(f"ID: {document['id']}, "
                  f"Status: {document['status']},"
                  f"Prompt: {document['prompt']}",
                  f"Time: {document['time']}",
                  f"Url: {document['url']}")
        print("-------------------------------------------------------------------")
        print(collection.count_documents({}))
        prompt = input("Enter your request id for delete: ")
        collection.delete_one({"id": int(prompt)})
        print("Well done! Your deleted one photo!")
        print("-------------------------------------------------------------------")

    else:
        numbers_documents = random.randint(1, 1000000)
        url_file = document # enter url of photo from openai(Нужно подогнать на Вашей части кода сгенерированый url в
        #переменную и присвоить ее этой переменной)
        data = [{
            "id": numbers_documents,
            "status": "The link is not active after 2 hours from the moment of generation",
            "prompt": PROMPT,
            "url": url_file,
            "time": datetime.datetime.now()}]
        collection.insert_many(data)
        print("Well done! Your added one photo!")


except Exception as error:
    print(error)
    print('Not connected to MongoDB')

















































