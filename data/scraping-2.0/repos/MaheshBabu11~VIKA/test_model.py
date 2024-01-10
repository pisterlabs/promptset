from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import openai
openai.api_key = ""

def predict_model(query):
    model =load_model("my_model.h5")
    vec = CountVectorizer()
    xeval=list(query)
    xeval_numeric = vec.fit_transform(xeval).toarray() 
    prediction=model.predict(xeval_numeric)
    y_pred=np.where(prediction>=0.5,1,0)
    if y_pred==0:
        return "Negative"
    else:
        return "Positive"


# place your openai beta key here.


def predict_gpt3(query):

	response = openai.Completion.create(
  	engine="davinci",
  	prompt="This is a tweet sentiment classifier\nTweet: \"I loved the new Batman movie!\"\nSentiment: Positive\n###\nTweet: \"I hate it when my phone battery dies\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet text\n\n\n1. \"I loved the new Batman movie!\"\n2. \"I hate it when my phone battery dies\"\n3. \"My day has been 👍\"\n4. \"This is the link to the article\"\n5. \"This new music video blew my mind\"\n\n\nTweet sentiment ratings:\n1: Positive\n2: Negative\n3: Positive\n4: Neutral\n5: Positive\n\n\n###\nTweet text\n\n\n1. \"I can't stand homework\"\n2. \"This sucks. I'm bored 😠\"\n3. \"I can't wait for Halloween!!!\"\n4. \"My cat is adorable ❤️❤️\"\n5. \"I hate chocolate\"\n\n\nTweet sentiment ratings:\n1.i am sad\n2.i am sad\n3.i am sad\n4.i am sad\n5.i am sad\n\n\nI hate life\n###\n\nTweet: \"I hate life\"\nSentiment: Negative\n"+query+"###",
  	temperature=0.3,
  	max_tokens=60,
  	top_p=1,
  	frequency_penalty=0,
  	presence_penalty=0,
  	stop=["###"]
		)
	val=response['choices'][0]['text'].split()
	for i in val:
		if i=="Positive" or i =="Negative":
			return i


def code_mode(query):
	start_sequence = "\nAI:"
	restart_sequence = "\nHuman: "

	response = openai.Completion.create(
  engine="davinci",
  prompt="Q: Ask Constance if we need some bread\nA: send-msg `find constance` Do we need some bread?\nQ: Send a message to Greg to figure out if things are ready for Wednesday.\nA: send-msg `find greg` Is everything ready for Wednesday?\nQ: Ask Ilya if we're still having our meeting this evening\nA: send-msg `find ilya` Are we still having a meeting this evening?\nQ: Contact the ski store and figure out if I can get my skis fixed before I leave on Thursday\nA: send-msg `find ski store` Would it be possible to get my skis fixed before I leave on Thursday?\nQ: Thank Nicolas for lunch\nA: send-msg `find nicolas` Thank you for lunch!\nQ: Tell Constance that I won't be home before 19:30 tonight — unmovable meeting.\nA: send-msg `find constance` I won't be home before 19:30 tonight. I have a meeting I can't move.\nQ: python code to find the sum of 2 numbers\nA: sum=0; for i in range(1,11): sum=sum+i print sum\nQ:python code to find the sum of any two given numbers\nA: sum=0; for i in range(1,11): sum=sum+i print sum\nQ:python code to find the sum of any two given numbers\n\nA: sum=0; for i in range(1,11): sum=sum+i print sum\n\nQ: python code to find the largest of 2 numbers\nA: if i>j: k=i print k else: k=j print k\nQ:"+query+"\n",
  temperature=0.89,
  max_tokens=500,
  top_p=1,
  frequency_penalty=0.2,
  presence_penalty=0,
  stop=["\n"]
)
	val=response['choices'][0]['text'].replace("A:","")
	return val


