import openai
import time

openai.api_key ="[insert open ai token here]"

def callChatAPI(content,msgs=None,temperature=1):
    if msgs is None:    
        msgs=[            
          {
              "role": "user", "content":content          
          }
        ]
    response=openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0,    
    messages=msgs
    )
    return response

def callDavinci(content):    
    response=openai.Completion.create(
      model = "text-davinci-002",
      prompt= content,
      temperature=0.0,
      max_tokens=100, 
      n= 1,
    )
    return response

def retry_callAPI(content,msgs=None,api='chat'):
    retry=0
    response=None
    while retry<2:
        try:
#             response=callAPI(content,msgs)
            if(api=='chat'):
                response=callChatAPI(content,msgs)
            else:    
                response=callDavinci(content,msgs)
            break
        except:
            retry+=1
            time.sleep(3)
    return response 