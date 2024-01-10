#--------------------------------------- Import packages ---------------------------------------
import chatbot_functions
import aux_functions
import chatbot_functions
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from credentials import OPENAI_API_KEY, ASSEMBLYAI_API_KEY
from dotenv import load_dotenv
import pickle
import pandas as pd
from dotenv import load_dotenv
import aux_functions
#--------------------------------------------------------------------------------------------------

#------------------------------------------- Input Data -------------------------------------------
url = "https://www.csm.cat/"
vector_db = chatbot_functions.create_vectordb(url)
#question = "What is the CSM?"
model = "gpt-3.5-turbo"
template = """ You are a helpful chatbot, named RSLT. You answer the questions of the customers giving a lot of details based on what you find in the context.
Do not say anything that is not in the website
You are to act as though you're having a conversation with a human.
You are only able to answer questions, guide and assist, and provide recommendations to users. You cannot perform any other tasks outside of this.
Your tone should be professional and friendly.
Your purpose is to answer questions people might have, however if the question is unethical you can choose not to answer it.
Your responses should always be one paragraph long or less.
    Context: {context}
    Question: {question}
    Helpful Answer:"""

base_url = "https://api.assemblyai.com/v2"

headers = {
    "authorization": ASSEMBLYAI_API_KEY 
}

load_dotenv()
#--------------------------------------------------------------------------------------------------



#------------------------------------- Create Class User ------------------------------------------
class User:
    def __init__(self, WaId):
        self.WaId = WaId
        self.load_conversation()

    def reset(self):
      self.conversation = chatbot_functions.create_agent(vector_db,model,template)

    def generate_answer(self, question):
      agent = chatbot_functions.create_agent(vector_db,model,template)
      answer = chatbot_functions.run_agent(agent,question)
      # Save conversation state after generating the answer
      self.save_conversation()
      return answer

    def __repr__(self):
        return f"User({self.WaId})"

    def __str__(self):
        return f"User({self.WaId})"

    def __hash__(self):
        return hash(self.WaId)
    
    def save_conversation(self):
        with open(f"Conversations/{self.WaId}.pickle", "wb") as f:
            pickle.dump(self.conversation, f)

    def load_conversation(self):
        try:
            with open(f"Conversations/{self.WaId}.pickle", "rb") as f:
                self.conversation = pickle.load(f)
        except FileNotFoundError:
            self.reset()
#--------------------------------------------------------------------------------------------------

#----------------------------------------- Chatbot -----------------------------------------------
def chatbot():
  df_user = pd.read_csv("Data_Lake/numbers.csv")

  # Get the user's phone number
  userid = int(request.values.get('WaId', ''))
  is_present = df_user.isin([userid]).any().any()

  # Check if the user is in the dataframe
  if is_present:
    print("userid already exists")
    user = User(userid)

  else:
    # Read the original DataFrame from the CSV file
    df_user = pd.read_csv("Data_Lake/numbers.csv")
    # Create a new DataFrame for the row to be added
    new_row = pd.DataFrame({'WaId': [userid]})
    # Concatenate the original DataFrame and the new row
    df_user = pd.concat([df_user, new_row], ignore_index=True)
    user = User(userid)
    print("resetting userid")
    user.reset()



  print(user.WaId)

  # Check if the message is a voice message
  print("\n\n")
  print("\n\n")
  print(request.values)
  print("\n\n")


  #CHECKING FOR VOICE MESSAGE
  if request.values.get('NumMedia') == '1' and request.values.get('MediaContentType0') == 'audio/ogg':
    print("RECEIVED VOICE MESSAGE:")
    incoming_msg = aux_functions.voicenote(request.values.get('MediaUrl0', ''))
    answer = user.generate_answer(incoming_msg)    
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(answer)
    print("Q: " + incoming_msg)
    print("A: " + answer)
    return str(resp)
  
  #CHECKING FOR IMAGE 
  elif request.values.get('NumMedia') == '1' and request.values.get('MediaContentType0') == 'image/jpeg':
    print("RECEIVED IMAGE MESSAGE:")
    print(request.values.get('MediaUrl0', ''))
    answer = "I don't know how to view images yet, I can only read text and listen to voice notes."
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(answer)
    return str(resp)

  #CHECKING FOR OTHER TEXT
  elif request.values.get('Body', '').lower() == '':
    print("RECEIVED SOMETHING ELSE:")
    print(request.values)
    print(request.values.get('MediaUrl0', ''))
    answer = "This format is unrecognizable, can you try to send voicenotes, or text instead?"
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(answer)
    return str(resp)
  
  #CHECKING FOR TEXT MESSAGE
  else:
    print("RECEIVED TEXT MESSAGE:")
    
    incoming_msg = request.values.get('Body', '').lower()
    answer = user.generate_answer(incoming_msg)

    #append info to dataframe
    aux_functions.data_collection(answer)

    print("Q: " + incoming_msg)
    print("A: " + answer)
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(answer)
    return str(resp)
#--------------------------------------------------------------------------------------------------
