#constants
import os
import openai

# def Get_API():
#     #API_Key = "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"
#     set OPENAI_API_KEY = "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"
#     return #API_Key
def Get_API():
  os.environ["OPENAI_API_KEY"] = "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"
  return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

api_client = Get_API()
def Get_Preface():
    Preface = 'Add frequent punctuation and line breaks to optimize you text response for translation to voice using gTTs which has a 100 character limit. Construct your responses with a style of communication that implies that you are a close buddy of mine.  Use Casual Tone: Skip the stiff formalities and maintain a light, humorous demeanor. My Question is : '
    return Preface
def Get_NewsAPI():
    AINewsToken =   "09d5d7423ae24e96a8d889cfb1b537bc" #"a0501547480f4cb69e4c0ba4fb96c17a"
    return AINewsToken
