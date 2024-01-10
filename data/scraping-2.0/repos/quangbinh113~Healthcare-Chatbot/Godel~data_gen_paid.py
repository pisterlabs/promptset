import os
import time
import openai
import json
import random
import argparse

from dotenv import load_dotenv, find_dotenv


root = os.getcwd()

_ = load_dotenv(find_dotenv('key.env')) # read local .env file

parser = argparse.ArgumentParser(description = "Kaggle-version to gendata")
parser.add_argument('--doc_start', default = 0, type = int ,help = 'the start document to use to gendata')
parser.add_argument('--doc_end', default = 1005, type = int ,help = 'the end document to use to gendata')
parser.add_argument('--access_id', default = 1, type = int ,help = 'Access to auth of chat GPT')
args = parser.parse_args()


#access_token get on https://chat.openai.com/api/auth/session
#replace your access_token on link above to key.env file
#use for free ChatGPT
access_token = os.getenv(f'ACCESS_TOKEN_{str(args.access_id)}')

openai.api_key = access_token
# Here is my key
# access_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJuaGF0bmFtMDQwNDIwMDJAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWV9LCJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsidXNlcl9pZCI6InVzZXItTmJuUFo3WXBnQTMxVWQ2cFNrQ3FTck5YIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDExNDgwNDA1MTY3MTIyODQzMzgxNSIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE2ODczNjIyOTAsImV4cCI6MTY4ODU3MTg5MCwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvcmdhbml6YXRpb24ud3JpdGUifQ.zRQmxzaFOLz5Aj2ProslaCxKcJVt-xtfI-JEABOHqUgYU8Eg56uDwukkHAnykUUUtW3jsZIRSYgBesNNinD4znRnkmYyvbqTTtTGLMFuUElD_EvM2bg80WSxIAdnXFSHgGD1-EU6xQoQwGeMqojHis9tSWCwZj1riCc2GQAnOR-PlMdXO5NK664HNcE6AAyS8XDk16ZFsqfa-zplG94ajtVzE3Q1MK_z8mj3oNT0cl06SBqt5i7V4vPspBXELuaU60Iolb-a-jS-RtQplU2AEXdpXFgh_TiY3IwFA7pSPMTG_RVPzNvpJg5Vj_EFPGmZCM6dGbSB70n7CqUgg55H0A"

# chatbot = Chatbot(config={"access_token": access_token,\
#                           "model": "gpt-3.5-turbo"
# })

def extract_json(text):
  s =0
  e = 0
  for i in range(len(text)):
    if text[i] == '[':
      s = i
      break
  for i in range(len(text)):
    if text[i] == ']':
      e = i
  data = []
  for da in eval(text[s:e+1]):
    da['dialog'] = da['dialog'][:9]
    data.append(da)
  return data

def get_answer(question):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
      {"role": "system", "content": "You are my AI bot to generate conversation between a chatbot and a patient. You produce each response in the character's perspective."},
      {"role": "user", "content": question}
    ],
    n = 10,
    temperature = 0.8
  )
  # for data in chatbot.ask(question,auto_continue=True):
  #     response = data["message"]
  # chatbot.reset_chat()
  # chatbot.delete_conversation(chatbot.get_conversations(0,1)[0]['id'])
  response = []
  for answer in completion.choices:
    try:
       response += extract_json(answer.message['content'])
    except Exception:
       with open("/kaggle/working/invalid.txt","a") as f :
            f.write(str(answer.message['content'])+'/n') 
       continue
  return response
  
# def collect_data(prompt,datasize = 30000):
#   count = 0
#   result = []
#   while True:
#     if count > datasize:
#       break
#     try:
#       result += extract_json(get_answer(prompt))
#       count+=1
#       time.sleep(2)
#     except SyntaxError:
#       time.sleep(2)
#     except Error as e:
#       if str(e) == "OpenAI: {\"detail\":\"You've reached our limit of messages per hour. Please try again later.\"} (code: 429)": 
#         time.sleep(720)
#       else:
#         print(str(e))
#         time.sleep(10)
#     except Exception as b:
#         print("Unknown: " + str(b))
#         continue
#   return result

def create_prompt(context):  
  return f"""
        I want you to generate multiple conversations related to a given context. The conversation is between the patient and the health-care chatbot called Medi. The conversation may be about one of following cases or can combine them or based on your imagination:
            The patient has some symptom in the context and needs help, Medi will give advice and diagnosis based on its knowledge obtained from the context.
            The patient just has some questions about the disease mentioned in the context and Medi gives the explanation for these questions.
            the patient has endured a medical procedure in the given context, Medi will give advice for lifestyle, diet, physical excerise based on its knowledge obtained from the context.
        You must write the output conversation in a following JSON format:

        {{
        "dialog": [patient's turn,
                   Medi's turn,
                   patient's turn,
                   Medi's turn,
                   patient's turn,
                   Medi's turn,
                   patient's turn,
                   Medi's turn,
                   last patient's turn],
        "knowledge": {context},
        "response": last Medi's turn
        }}

        The number of words in each turn of Medi and patient must not exceed 15 words no matter what. The conversation need not contain all the information in the context and should be as natural as possible. The replies of Medi and the patient must be simple and comprehensive for each other as much as possible.
        The last patient's turn must be a question relating to the given context and the response is the answer for last Medi's turn based on the given context 
        The dialog must contain 4 turns between Medi and patient

        The context is: #### {context} ####
        
        From the context, you must output a list of 3 dialogues in the following format:

        [dialogue 1, 
        dialogue 2,
        dialogue 3,
        ]
        
        This is an example of output: 

        [
        {{
        "dialog": ["Hello Medi, I have curiosity about diabetes type 2.",
                   "No problem. I am right here to answer your question as a chatbot medical supporter",
                   "Firstly, I want to know about the real state of affairs of this type of the disease",
                   "This type diabetes is the most common type of diabetes mellitus accounting for 95% of diabetes. ",
                   "All right, another question, how can we prevent to transformming process from",
                   "We can prevent this process by lifestyle changes or medications that improve insulin sensitivity or reduce the liver's glucose production",
                   "About medications, insulin is a good choice, right ?",
                   "It depends on how high your blood sugar is and what your doctor suggests, my advice is that you should modify your lifestyle first.",
                   "Interesting. But now I wonder which causes diabetes type 2 ?"],
        "knowledge": "Diabetes Diabetes  The defective responsiveness of body tissues to insulin is believed to involve the insulin receptor.  However, the specific defects are not known.  Diabetes mellitus cases due to a known defect are classified separately.  Type 2 diabetes is the most common type of diabetes mellitus accounting for 95% of diabetes.  Many people with type 2 diabetes have evidence of prediabetes  before meeting the criteria for type 2 diabetes.  The progression of prediabetes to overt type 2 diabetes can be slowed or reversed by lifestyle changes or medications that improve insulin sensitivity or reduce the liver's glucose production. Type 2 diabetes is primarily due to lifestyle factors and genetics Diabetes Diabetes",
        "response": "As far as i know, the two main reasons lead to diabetes type 2 are lifestyle factors and genetics"
        }},
        {{
        "dialog": ["Hello Medi, I feel not well today, can you help me ?",
                   "Sure, It's my pleasure, so can you tell me the symptoms you got?",
                   "Of course, I have a headaches in the morning, and feel very uncomfortable and very tired. I can not feel the taste of my lunch.My throat is terrible and my nose is running.",
                   "Oke i got it. Next, how long have you suffered these symptoms ?",
                   "I have to tolerate these all day long. It started in the morning",
                   "So yesterday did you do somthing subnormal ?",
                   "I am quite sure it's no, I just drink some ice cold although these dáy it's cold. ",
                   "I see, maybe that is the reason. ",
                   "So what is wrong with me, Medi?"]
        "knowledge": "Influenza Influenza  The onset of symptoms is sudden, and initial symptoms are predominately non-specific, including fever, chills, headaches, muscle pain or aching, a feeling of discomfort, loss of appetite, lack of energy/fatigue, and confusion.  These symptoms are usually accompanied by respiratory symptoms such as a dry cough, sore or dry throat, hoarse voice, and a stuffy or runny nose.  Coughing is the most common symptom.  Gastrointestinal symptoms may also occur, including nausea, vomiting, diarrhea, and gastroenteritis, especially in children.  The standard influenza symptoms typically last for 2 - 8 days.   A 2021 study suggests influenza can cause long lasting symptoms in a similar way to long COVID. Symptomatic infections are usually mild and limited to the upper respiratory tract, but progression to pneumonia is relatively common Influenza Influenza",
        "response": "Based on all your symptoms such as fever, headaches and tired as well as loss of appetite, specially running nose and sore throat, you get Influenza."
        }},
        {{
        "dialog": ["Medi, I need your advice about my life style. I had obesity long time ago.",
                   "My advice on obersity is that you should change your lifestyle",
                   "Can you give me some examples?",
                   "Simple thing like eating less sugar, do more physical exercises and balance your diet with more vitamins and vegetables",
                   "Those are very helpful advice. However are there many people out there suffering from obesity?",
                   "There are more than 1 billion people having obesity which is a very large amount",
                   "Does the gorverment or society have any action to prevent the increase of this situation",
                   "Yes, they have",
                   "Oh, so what are they?"],
        "knowledge": "Obesity Obesity Obesity is a complex public health and policy problem because of its prevalence, costs, and health effects.  As such, managing it requires changes in the wider societal context and effort by communities, local authorities, and governments.  Public health efforts seek to understand and correct the environmental factors responsible for the increasing prevalence of obesity in the population.  Solutions look at changing the factors that cause excess food energy consumption and inhibit physical activity.  Efforts include federally reimbursed meal programs in schools, limiting direct junk food marketing to children, and decreasing access to sugar-sweetened beverages in schools.  The World Health Organization recommends the taxing of sugary drinks.  When constructing urban environments, efforts have been made to increase access to parks and to develop pedestrian routes Obesity Obesity",
        "response": "Many action have been taken such as seeking to understand and correct the environmental factors responsible for the increasing prevalence of obesity in the population, changing the factors that cause excess food energy consumption and inhibit physical activity, federally reimbursed meal programs in schools, limiting direct junk food marketing to children, and decreasing access to sugar-sweetened beverages in schools, taxing of sugary drinks or increasing access to parks and to develop pedestrian routes"
        }}
        ]

        The example is not the final output
        """

import argparse

parser = argparse.ArgumentParser(description='Transformation my dataset to group dataset type')
parser.add_argument('--save_file', type=str,default="",
                    help='transformation saving file path ')
args = parser.parse_args()
if __name__ == "__main__":
  
  for i,filepath in enumerate( sorted(list( os.listdir(os.path.join(root,"chunking")) ) )[args.doc_start:args.doc_end] ):
      try:
        with open(os.path.join(root,"chunking",filepath),"r") as fb:
            data = get_answer(create_prompt(fb.read()))
        with open(args.save_file + f"/medical_full_{str(i+args.doc_start)}.json","w") as fc:
            json.dump(data,fc, indent = 4)
        time.sleep(15)
      except Exception as e:
        print(e)
        break
  