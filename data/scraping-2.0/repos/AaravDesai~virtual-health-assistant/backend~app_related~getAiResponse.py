import sys
from db import db
from bson import ObjectId
from flask import make_response
import openai

openai.api_type = "azure"
openai.api_key = '9fd2d1a84636415685bcf7dc451040fb'
openai.api_base = 'https://api.umgpt.umich.edu/azure-openai-api/ptu'
openai.api_version = '2023-03-15-preview'

def getAiResponse_api(params):
    session = db["session"]
    users_collection = db["users"]
    
    sessionId = params["sessionId"]
    user = users_collection.find_one({"_id": ObjectId(params["cookie"])})
    initial = user['diet_plan']+user["workout_plan"]
    
    conversation_history = [
        {"role": "system", "content": "You are a helpful virtual health assistant"},
        {"role": "assistant", "content": f"Here are some Recommendations: {initial}"},
    ]
    
    if sessionId=='':
        new_session={
            "user":params["cookie"],
            "conversation":[params["message"]]
        }
        inserted_session = session.insert_one(new_session)
        sessionId = str(inserted_session.inserted_id)
    else:
        curr_session = session.find_one({"_id": ObjectId(sessionId)})
        for i in range(len(curr_session["conversation"])):
            if i%2==0:
                conversation_history.append({"role": "user", "content": curr_session["conversation"][i]})
            else:
                conversation_history.append({"role": "assistant", "content": curr_session["conversation"][i]}) 
    print(conversation_history,file=sys.stderr)
    conversation_history.append({"role": "user", "content": params["message"]})    
    
    response = openai.ChatCompletion.create(
        engine='gpt-4',
        messages=conversation_history
    )
    assistant_reply = response['choices'][0]['message']['content']  
    session.update_one({"_id": ObjectId(sessionId)},{"$push":{'conversation':assistant_reply}})
    
    return make_response({"message":assistant_reply,"sessionId":sessionId})