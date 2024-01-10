from pymongo.mongo_client import MongoClient
from bson.objectid import ObjectId
import openai

# Connection details
URI = "mongodb+srv://VenkatSagi:mongodb.2004@cluster0.ijkgw8w.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(URI)
dataBase = client['ToyotaRaces']
collection = dataBase['DragStripRaces']

# API Key
openai.api_key = "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"

# Mongo ID & Info
raceID = "6546b55174909bb23005250a"
raceInfo = collection.find_one({'_id': ObjectId(raceID)})
vehicleModel = raceInfo['Vehicle']

# vv Example data set from Database vv
'''
raceInfo = {
    "_id":{"$oid":"6546b55174909bb23005250a"},
    "Date":"02-06-2023",
    "Time":"07:00 AM - 07:01 AM",
    "Vehicle":"Toyota GR Yaris",
    "Location of Race":"York Raceway",
    "Total Race Time":"00:10:250",
    "Reaction Time":"00:00:600",
    "60-foot time":"00:01:600",
    "330-foot time":"00:04:100",
    "660-foot time":"00:06:100",
    "1/8-mile speed":"85 mph",
    "1,000-foot time":"00:08:200",
    "1/4-mile time":"00:10:250",
    "1/4-mile speed":"130 mph",
    "MPH":"140"
}
'''

# ChatGPT Recomendations
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system",
         "content": f"Provides recomendations for the user, based on their Drag Strip Time slip info provided."
                    f"Gives specific recomendations to improve performance."
                    f"Recomends what the user need the most improvement on (analize time slip stats)."
                    f"Specifies how (be specific on how) they can improve it."
                    f"Give recomendations of potential car parts, modifications, drills, etc."
                    f"Mentions the user's (before) stats and tells them potential (after) stats with specific improvements."
                    f"Tells them how much time they can save with specific modifications, drills, etc."
                    f"Gives result in bullet point format."
                    f"Time Slip & Data: {raceInfo}"},
        {"role": "user", "content": f"Give me specific recomendations to improve performance for my {vehicleModel}"}
    ]
)  # open ai compltetions api
print(completion.choices[0].message.content)  # prints out Open Ai response