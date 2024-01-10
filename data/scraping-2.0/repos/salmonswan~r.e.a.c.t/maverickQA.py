from qa.bot import GroundedQaBot
import requests
import json
import os
from qdrant_client import QdrantClient
import cohere
co = cohere.Client(os.environ['mavqa'])

qdrant_client = QdrantClient(
  host="81c38949-2ca4-400c-b2b1-eb6041356987.us-east-1-0.aws.cloud.qdrant.io",
  api_key=os.environ['qdrant'],
)

cohere_api_key = os.environ['mavqa']
serp_api_key = 0
# serp_api_key = os.environ['serp']
rapidapi_key = os.environ['rapid']

bot = GroundedQaBot(cohere_api_key, serp_api_key)

def get_language(payload, api_key):
  url = "https://translate-plus.p.rapidapi.com/language_detect"

  payload = {"text": payload}
  headers = {
    "content-type": "application/json",
    "X-RapidAPI-Key": api_key,
    "X-RapidAPI-Host": "translate-plus.p.rapidapi.com"
  }

  response = requests.request("POST", url, json=payload, headers=headers)
  return json.loads(response.text)["language_detection"]["language"]

#translate
def translate(payload, api_key, detected_lang='en', to_en=True):
  if to_en:
    payload = {
      "langpair": f"{detected_lang}|en",
      "q": payload,
      "mt": "1",
      "onlyprivate": "0",
      "de": "a@b.c"
    }
  else:
    payload = {
      "langpair": f"en|{detected_lang}",
      "q": payload,
      "mt": "1",
      "onlyprivate": "0",
      "de": "a@b.c"
    }
  url = "https://translated-mymemory---translation-memory.p.rapidapi.com/get"
  headers = {
    "X-RapidAPI-Key": rapidapi_key,
    "X-RapidAPI-Host":
    "translated-mymemory---translation-memory.p.rapidapi.com"
  }

  response = requests.request("GET", url, headers=headers, params=payload)

  return json.loads(response.text)["responseData"]["translatedText"]

###required for conversant

#Everything in farmer_config is editable to do prompt engineering
# typhoid_config = {
#   "preamble":
#   "Below is a conversation between a typhoid expert and a person.",
#   "example_separator":
#   "<CONVERSATION>\n",
#   "headers": {
#     "user": "person",
#     "bot": "Typhoid Expert",
#   },
#   "examples": [[
#     {
#       "user":
#       "What is typhoid fever?",
#       "bot":
#       "Typhoid fever is a bacterial infection caused by Salmonella typhi. It can be transmitted through contaminated food or water, and symptoms can include fever, general ill-feeling, and abdominal pain.",
#     },
#     {
#       "user":
#       "How is typhoid fever treated?",
#       "bot":
#       "Typhoid fever is typically treated with antibiotics, and you may need to stay in the hospital for a few days to receive intravenous antibiotics. It’s important that you get plenty of rest and stay hydrated as well. ",
#     },
#     {
#       "user":
#       "How do I avoid getting Typhoid fever?",
#       "bot":
#       "Make sure to wash your hands frequently and avoid preparing food for others until you have been cleared by a doctor. It’s also important to avoid sharing utensils, cups, or other personal items.",
#     },
#   ]],
# }

def MaverickQA(query):
  translating = True
  if translating:
    bot = GroundedQaBot(cohere_api_key, serp_api_key)
    verbosity = 0
    print("querying")
    question = query
    detected_lang = get_language(query, rapidapi_key)
    if detected_lang != 'en':
      question = translate(query, rapidapi_key, detected_lang, True)
    print(question)
    # reply, source_urls, source_texts = bot.answer(question,
    #                                               verbosity=verbosity,
    #                                               n_paragraphs=5)
    reply, source_texts = bot.answer(question,
                                     verbosity=verbosity,
                                     n_paragraphs=5)
    
    if detected_lang != 'en':
      reply = translate(reply, rapidapi_key, detected_lang, False)
  else: 
    bot = GroundedQaBot(cohere_api_key, serp_api_key)
    verbosity = 0
    print("querying")
    question = query
    reply, source_texts = bot.answer(question,
                                     verbosity=verbosity,
                                     n_paragraphs=5)
  #   typhoid_bot = conversant.PromptChatbot(
  #   client=co,
  #   prompt=ChatPrompt.from_dict(typhoid_config),
  #   chatbot_config={'model': 'command-xlarge-nightly'})
  #   return typhoid_bot.reply(query)


  # print(reply)

  # # sources_str = "\n".join(list(set(source_urls)))
  # # reply_incl_sources = f"{reply}\nSource:\n{sources_str}"
  return reply


# def MaverickQA(query):
#   response = co.embed(texts=[query], model="large")
#   embeddings = response.embeddings[0]
#   embeddings = [float(i) for i in embeddings]

#   search_result = qdrant_client.search(collection_name="mycollection",
#                                        query_vector=embeddings,
#                                        limit=5)
#   print(search_result)

#   # prompt = "Answer the question given the following context:\n"
#   # for result in search_result:
#   #     for i in result.payload['text']:
#   #         prompt += i
#   #     prompt += "\n---\n"
#   # prompt += "Question:" + query + "\n---\n" + "Answer:"

#   #changed the prompt to better reflect the question

#   prompt = "This is an example of question answering based on a text passage:\nContext:  "
#   for result in search_result:
#     for i in result.payload['text']:
#       prompt += i
#     prompt += "\n---\n"
#   prompt += "Question:" + query + "\n---\n" + "Answer:"

#   print(prompt)

#   print("----PROMPT START----")
#   print(":", prompt)
#   print("----PROMPT END----")

#   response = co.generate(model='command-xlarge-nightly',
#                          prompt=prompt,
#                          max_tokens=300,
#                          temperature=0.9,
#                          k=0,
#                          p=0.75,
#                          stop_sequences=['---'],
#                          return_likelihoods='NONE',
#                          logit_bias={'11': -10},
#                          num_generations=5)

#   return response.generations[0].text
