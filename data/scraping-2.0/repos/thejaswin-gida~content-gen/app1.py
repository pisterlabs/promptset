# import gradio as gr
# from gradio import Dropdown, Textbox, Textbox, Textbox
# import os
# # from langchain.llms import OpenAI  #uncomment to use stream function and comment other openai import
# from langchain.chat_models import ChatOpenAI, ChatAnthropic
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.schema import HumanMessage
# from langchain.callbacks.base import BaseCallbackHandler
# from threading import Thread
# from queue import Queue, Empty
# from threading import Thread
# from collections.abc import Generator
# from langchain.callbacks.base import BaseCallbackHandler
# from langchain.chains import ConversationChain
# from langchain.prompts import PromptTemplate
# import openai
# import json
# import requests
# import time
# import vertexai
# from vertexai.preview.language_models import TextGenerationModel
# from deep_translator import GoogleTranslator
# from cryptography.fernet import Fernet
# import tempfile
# from langchain.agents import Tool
# from langchain.agents import AgentType
# from langchain.agents import initialize_agent
# from langchain.tools import DuckDuckGoSearchRun
# from openai import OpenAI
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize
# import concurrent.futures

# language_codes = {'Afrikaans': 'af','Albanian': 'sq','Amharic': 'am','Arabic': 'ar','Armenian': 'hy','Assamese': 'as','Aymara': 'ay','Azerbaijani': 'az','Bambara': 'bm','Basque': 'eu','Belarusian': 'be','Bengali': 'bn','Bhojpuri': 'bho','Bosnian': 'bs','Bulgarian': 'bg','Catalan': 'ca','Cebuano': 'ceb','Chinese': 'zh-TW','Corsican': 'co','Croatian': 'hr','Czech': 'cs','Danish': 'da','Dhivehi': 'dv','Dogri': 'doi','Dutch': 'nl','English': 'en','Esperanto': 'eo','Estonian': 'et','Ewe': 'ee','Filipino': 'fil','Tagalog': 'fil','Finnish': 'fi','French': 'fr','Frisian': 'fy','Galician': 'gl','Georgian': 'ka','German': 'de','Greek': 'el','Guarani': 'gn','Gujarati': 'gu','Haitian Creole': 'ht','Hausa': 'ha','Hawaiian': 'haw','Hebrew': 'he','Hindi': 'hi','Hmong': 'hmn','Hungarian': 'hu','Icelandic': 'is','Igbo': 'ig','Ilocano': 'ilo','Indonesian': 'id','Irish': 'ga','Italian': 'it','Japanese': 'ja','Javanese': 'jv','Kannada': 'kn','Kazakh': 'kk','Khmer': 'km','Kinyarwanda': 'rw','Konkani': 'gom','Korean': 'ko','Krio': 'kri','Kurdish': 'ku','Kurdish': 'ckb','Sorani': 'ckb','Kyrgyz': 'ky','Lao': 'lo','Latin': 'la','Latvian': 'lv','Lingala': 'ln','Lithuanian': 'lt','Luganda': 'lg','Luxembourgish': 'lb','Macedonian': 'mk','Maithili': 'mai','Malagasy': 'mg','Malay': 'ms','Malayalam': 'ml','Maltese': 'mt','Maori': 'mi','Marathi': 'mr','Meiteilon': 'mni-Mtei','Manipuri': 'mni-Mtei','Mizo': 'lus','Mongolian': 'mn','Myanmar': 'my','Burmese': 'my','Nepali': 'ne','Norwegian': 'no','Nyanja': 'ny','Chichewa':'ny','Odia': 'or','Oriya': 'or','Oromo': 'om','Pashto': 'ps','Persian': 'fa','Polish': 'pl','Portuguese': 'pt','Punjabi': 'pa','Quechua': 'qu','Romanian': 'ro','Russian': 'ru','Samoan': 'sm','Sanskrit': 'sa','Scots Gaelic': 'gd','Sepedi': 'nso','Serbian': 'sr','Sesotho': 'st','Shona': 'sn','Sindhi': 'sd','Sinhala': 'si','Slovak': 'sk','Slovenian': 'sl','Somali': 'so','Spanish': 'es','Sundanese': 'su','Swahili': 'sw','Swedish': 'sv','Tagalog': 'tl','Tajik': 'tg','Tamil': 'ta','Tatar': 'tt','Telugu': 'te','Thai': 'th','Tigrinya': 'ti','Tsonga': 'ts','Turkish': 'tr','Turkmen': 'tk','Twi': 'ak','Ukrainian': 'uk','Urdu': 'ur','Uyghur': 'ug','Uzbek': 'uz','Vietnamese': 'vi','Welsh': 'cy','Xhosa': 'xh','Yiddish': 'yi','Yoruba': 'yo','Zulu': 'zu'}

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# encryption_key = os.environ['enc_key']

# fernet = Fernet(encryption_key)

# # Decrypt the encrypted data
# with open("encrypted_data.json", "rb") as encrypted_file:
#     encrypted_data = encrypted_file.read()
#     decrypted_data = fernet.decrypt(encrypted_data)

# with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
#     temp_file.write(decrypted_data)
#     temp_file_path = temp_file.name

# # Set the temporary file path as the Google Cloud credentials
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path
# vertexai.init(project=os.environ['project_id'], location='us-central1')

# m,t,ans="","",""

# class QueueCallback(BaseCallbackHandler):
#     def __init__(self, q):
#         self.q = q

#     def on_llm_new_token(self, token: str, **kwargs: any) -> None:
#         self.q.put(token)

#     def on_llm_end(self, *args, **kwargs: any) -> None:
#         return self.q.empty()


# # def stream(input_text) -> Generator:
# #     q = Queue()
# #     job_done = object()


# #     """Logic for loading the chain you want to use should go here."""
# #     llm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
# #         streaming=True,
# #         model='gpt-4',
# #         callbacks=[QueueCallback(q)],
# #     )

# #     conversation = ConversationChain(
# #         prompt= PromptTemplate(
# #     input_variables=['history', 'input'],
# #     template=(
# #         "Imagine you are an AI Content Generator. Your goal is to create unique, engaging, and high-quality content based on the given input with an example. if language is sepcified your task is to generate content in the language specified by the user while keeping text within quotes in English (like this: ' we will rock you') and the rest in the chosen language. Use the appropriate devanagari script for the chosen language. If the user doesn't specify a language, default to English. Ensure that all sentences are grammatically correct. When handling language specification, make sure that any text within quotes remains in English while the rest of the content is in the chosen language. If the user provides text formatting instructions like lists, be sure to follow them. Also, ensure that the generated content contains no questions, and if a word count is specified, do not exceed it.  \n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
# #         )),
# #         llm=llm,
# #         verbose=True
# #     )
# #     # Create a funciton to call - this will run in a thread
# #     def task():
# #         resp = conversation.run(input_text)
# #         q.put(job_done)

# #     # Create a thread and start the function
# #     t = Thread(target=task)
# #     t.start()

# #     content = ""

# #     # Get each new token from the queue and yield for our generator
# #     while True:
# #         try:
# #             next_token = q.get(True, timeout=1)
# #             if next_token is job_done:
# #                 break
# #             content += next_token
# #             yield next_token, content
# #         except Empty:
# #             continue

# def stream(input_text):
#     llm = ChatOpenAI(model='gpt-4')

#     conversation = ConversationChain(
#         prompt=PromptTemplate(
#             input_variables=['history', 'input'],
#             template=(
#                 "Imagine you are an AI Content Generator. Your goal is to create unique, engaging, and high-quality content based on the given input with an example. if language is sepcified your task is to generate content in the language specified by the user while keeping text within quotes in English (like this: ' we will rock you') and the rest in the chosen language. Use the appropriate devanagari script for the chosen language. If the user doesn't specify a language, default to English. Ensure that all sentences are grammatically correct. When handling language specification, make sure that any text within quotes remains in English while the rest of the content is in the chosen language. If the user provides text formatting instructions like lists, be sure to follow them. Also, ensure that the generated content contains no questions, and if a word count is specified, do not exceed it.  \n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
#             )),
#         llm=llm,
#     )
#     return conversation.run(input_text)

# def predict_gpt4(input_text):
#     llm = ChatOpenAI(
#         model_name="gpt-4",
#         temperature=0.5,
#     )

#     DDGsearch = DuckDuckGoSearchRun()

#     PREFIX = """
#                 Imagine you are an AI Content Generator.
#                 Your goal is to create unique, engaging and descriptive content based on the given input with an example. 
#                 If a word count is specified in given input, Strictly maintain the word limit.
#                 Ensure that all sentences are grammatically correct. 
#                 If the user provides text formatting instructions like lists, be sure to follow them. 
#                 Avoid any url links in the content.
#                 Also, ensure that the generated content contains no questions.
#             """
#     SUFFIX = """
#     Begin!
#     Instructions: 
#     {input}
#     {agent_scratchpad}
#     """

#     tools = [
#         Tool(
#             name="Duck Duck Go Search Results Tool",
#             func=DDGsearch.run,
#             description="Useful for search for information on the internet",
#         ),
#     ]

#     runAgent = initialize_agent(
#         tools,
#         llm,
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         verbose=False,
#         agent_kwargs={"prefix": PREFIX, "suffix": SUFFIX},
#         handle_parsing_errors=True,
#     )

#     context = str(runAgent.run(input_text))

#     client = OpenAI()
#     conversation = [
#         {"role": "system", "content": "You are an AI language model."},
#         {
#             "role": "user",
#             "content": f""" Input: {input_text} context: {context}. \n\n\n
#                         Strictly Follow the Below Instructions: 
#                         1. Imagine you are an Content Generator.
#                         2. Avoid 'Input' in your content, avoid same text of context and questions in content. 
#                         3. Your goal is to create unique, engaging and descriptive content on your own for given input with an example having given context as additional knowledge .
#                         4. If a word count is specified in given input, Strictly maintain the word limit.
#                         5. Ensure that all sentences are grammatically correct. 
#                         6. If the user provides text formatting instructions like lists, be sure to follow them. 
#                         7. Also, ensure that the generated content contains no questions.
#                         8. Omit any references to the AI model.""",
#         }
#         # Try to avoid depending too heavily on the provided context. Try to create your own content.
#     ]
#     return (
#         client.chat.completions.create(model="gpt-4", messages=conversation)
#         .choices[0]
#         .message.content.replace("\n", "")
#     )

# def predict_palm2(
#     project_id: str,
#     model_name: str,
#     temperature: float,
#     max_decode_steps: int,
#     top_p: float,
#     top_k: int,
#     content: str,
#     location: str = "us-central1",
#     tuned_model_name: str = "",
#     ) :
#     """Predict using a Large Language Model."""
#     model = TextGenerationModel.from_pretrained(model_name)
#     if tuned_model_name:
#       model = model.get_tuned_model(tuned_model_name)
#     response = model.predict(
#         content,
#         temperature=temperature,
#         max_output_tokens=max_decode_steps,
#         top_k=top_k,
#         top_p=top_p,)
#     return response.text

# url = "https://api.perplexity.ai/chat/completions"

# def predict_llama2(text):
#   payload = {
#       "model": "llama-2-70b-chat",
#       "messages": [
#           {
#               "role": "system",
#               "content": "Imagine you are an AI Content Generator. strictly Avoid adding introductory sentence like 'Sure, here is some' "
#           },
#           {
#               "role":"system",
#               "content": "Ensure that all sentences are grammatically correct. If the user provides text formatting instructions like lists, be sure to follow them. "
#           },
#           {
#               "role": "system",
#               "content": "Also, ensure that the generated content contains no questions. Avoid greetings and conclusions."
#           },
#           {
#               "role": "system",
#               "content": "Strictly Avoid Incomplete Sentences. Try to complete the sentence."
#           },
#           {
#               "role": "user",
#               "content": text
#           }
#       ],
#   }
#   headers = {
#       "accept": "application/json",
#       "content-type": "application/json",
#       "authorization": os.environ['pplx_api_key']
#   }

#   response = requests.post(url, json=payload, headers=headers)

#   return json.loads(response.text)['choices'][0]['message']['content']

# # Define a function to generate content based on the selected model
# # def generate_content(selected_model, text, language=None):
# #     ans=''
# #     try:
# #         print(text)
# #         print(selected_model)
# #         print(language)
# #         if selected_model == 'GPT-4':
# #           ans = predict_gpt4(text)
# #           if not language:
# #             yield ans
# #           else:
# #             ans = GoogleTranslator(source='auto', target=language_codes[language]).translate(ans) 
# #             yield ans
    
# #         elif selected_model == 'Llama2':
# #           ans = predict_llama2(text)
# #           if not language:
# #             yield ans
# #           else:
# #             ans = GoogleTranslator(source='auto', target=language_codes[language]).translate(ans) 
# #             yield ans
    
# #         elif selected_model == 'PALM2':
# #           ans = predict_large_language_model_sample(os.environ['project_id'], "text-bison", 0.2, 512, 0.8, 40, f'Avoid input in your response. Imagine you are an AI Content Generator. Try to generate content for {text}.  if the word "article" is specified in input, then you try to add title to it otherwise strictly avoid adding title. If the user provides text formatting instructions like lists, be sure to follow them. Also, ensure that the generated content contains no questions, and if a word count is specified, do not exceed it. Your goal is to create unique, engaging, and high-quality content based on the given input.', "us-central1")
# #           if not language:
# #             yield ans
# #           else:
# #             ans = GoogleTranslator(source='auto', target=language_codes[language]).translate(ans)  
# #             yield ans
# #     except:
# #         pass
# #     finally:
# #         print(ans)

# def split_text(input_text, max_chunk_length=4995):
#     sentences = sent_tokenize(input_text)

#     # Initialize variables
#     chunks = []
#     current_chunk = ""

#     # Iterate through sentences
#     for sentence in sentences:
#         # Check if adding the sentence to the current chunk exceeds the max length
#         if len(current_chunk) + len(sentence) <= max_chunk_length:
#             current_chunk += sentence + " "
#         else:
#             # Save the current chunk and start a new one with the current sentence
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + " "

#     # Add the last chunk if it's not empty
#     if current_chunk:
#         chunks.append(current_chunk.strip())

#     return chunks

# def gtranslate(language, text):
#     return GoogleTranslator(source="auto", target=language_codes[language]).translate(text)

# def google_translation(language, text):
#     if len(text)>=5000:
#         print('Character length greater than 5000')
#         result_chunks = split_text(text)
#         language = [language]*len(result_chunks)
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#           result_chunks = list(executor.map(gtranslate, language, result_chunks))
#         return ' '.join(result_chunks)
#     return GoogleTranslator(
#                         source="auto", target=language_codes[language]
#                     ).translate(text)

# def postprocess(text):
#     return text.replace("\n\n", " ").replace("\r", "").replace("\n", " ")

# # Define a function to generate content based on the selected model
# def generate_content(selected_model, text, language=None):
#     global m,t,ans
#     selected_model=selected_model.split(" ")[0]
#     if selected_model==m and text==t:
#         print(text)
#         print(selected_model)
#         print(language)
#         ans = GoogleTranslator(source='auto', target=language_codes[language]).translate(ans) 
#         print(ans)
#         yield ans
#     else:
#         m=selected_model
#         t=text
#         try:
#             print(text)
#             print(selected_model)
#             print(language)
#             if selected_model == 'GPT-4':
#               ans = predict_gpt4(text)
#               if not language:
#                 yield ans
#               else:
#                 ans = google_translation(language, ans)
#                 yield ans
        
#             elif selected_model == 'Llama2':
#               ans = predict_llama2(text)
#               if not language:
#                 yield ans
#               else:
#                 ans = google_translation(language, postprocess(ans))
#                 yield ans
        
#             elif selected_model == 'PaLM2':
#               ans = predict_palm2(os.environ['project_id'], "text-bison", 0.2, 512, 0.8, 40, f'Avoid input in your response. Imagine you are an AI Content Generator. Try to generate content for {text}.  if the word "article" is specified in input, then you try to add title to it otherwise strictly avoid adding title. If the user provides text formatting instructions like lists, be sure to follow them. Also, ensure that the generated content contains no questions, and if a word count is specified, do not exceed it. Your goal is to create unique, engaging, and high-quality content based on the given input.', "us-central1")
#               if not language:
#                 yield ans.replace("**", "").replace("*", "")
#               else:
#                 ans = google_translation(language, postprocess(ans)).replace("**", "").replace("*", "")
#                 yield ans
#         except:
#             pass
#         finally:
#             print(ans)


# model_dropdown_choices = [
#     "GPT-4 ($0.006 + $0.008 / 100 output words)",
#     "PaLM2 ($0.0001 + $0.00006 / 100 output words)",
#     "Llama2 ($0.001 / requests)"
# ]

# # Create a dropdown for model selection
# model_dropdown = Dropdown(
#     label="Select Model",
#     choices= model_dropdown_choices
# )

# # Create other input components (text, language, word_count) as needed
# text_input = Textbox(label="Input Text")

# language_dropdown = Dropdown(label="Language", choices=['Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Aymara', 'Azerbaijani', 'Bambara', 'Basque', 'Belarusian', 'Bengali', 'Bhojpuri', 'Bosnian', 'Bulgarian', 'Catalan', 'Cebuano', 'Chinese', 'Corsican', 'Croatian', 'Czech', 'Danish', 'Dhivehi', 'Dogri', 'Dutch', 'Esperanto', 'Estonian', 'Ewe', 'Filipino', 'Tagalog', 'Finnish', 'French', 'Frisian', 'Galician', 'Georgian', 'German', 'Greek', 'Guarani', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi', 'Hmong', 'Hungarian', 'Icelandic', 'Igbo', 'Ilocano', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Kinyarwanda', 'Konkani', 'Korean', 'Krio', 'Kurdish', 'Sorani', 'Kyrgyz', 'Lao', 'Latin', 'Latvian', 'Lingala', 'Lithuanian', 'Luganda', 'Luxembourgish', 'Macedonian', 'Maithili', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Meiteilon', 'Manipuri', 'Mizo', 'Mongolian', 'Myanmar', 'Burmese', 'Nepali', 'Norwegian', 'Nyanja', 'Chichewa', 'Odia', 'Oriya', 'Oromo', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Quechua', 'Romanian', 'Russian', 'Samoan', 'Sanskrit', 'Scots Gaelic', 'Sepedi', 'Serbian', 'Sesotho', 'Shona', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tajik', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tigrinya', 'Tsonga', 'Turkish', 'Turkmen', 'Twi', 'Ukrainian', 'Urdu', 'Uyghur', 'Uzbek', 'Vietnamese', 'Welsh', 'Xhosa', 'Yiddish', 'Yoruba', 'Zulu'])

# # Create a Gradio interface
# gr.Interface(
#     fn=generate_content,
#     inputs=[model_dropdown, text_input, language_dropdown],
#     outputs=Textbox(label="Generated Content"),
#     title="AI Content Generator",
#     description="Your Ultimate AI Content Wizard! You can also specify the language using a dedicated dropdown field."
# ).queue().launch(debug=True,share=True)

import gradio as gr
from gradio import Dropdown, Textbox, Textbox, Textbox
import os
# from langchain.llms import OpenAI  #uncomment to use stream function and comment other openai import
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from threading import Thread
from queue import Queue, Empty
from threading import Thread
from collections.abc import Generator
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import openai
import json
import requests
import time
import vertexai
from vertexai.preview.language_models import TextGenerationModel
from deep_translator import GoogleTranslator
from cryptography.fernet import Fernet
import tempfile
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.tools import DuckDuckGoSearchRun
from openai import OpenAI
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import concurrent.futures
import boto3
import requests, uuid
from googleapiclient import discovery
from ftlangdetect import detect
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re

language_codes1 = {"afrikaans": "af","albanian": "sq","amharic": "am","arabic": "ar","armenian": "hy","assamese": "as","aymara": "ay","azerbaijani": "az","bambara": "bm","basque": "eu","belarusian": "be","bengali": "bn","bhojpuri": "bho","bosnian": "bs","bulgarian": "bg","catalan": "ca","cebuano": "ceb","chichewa": "ny","chinese (simplified)": "zh-CN","chinese (traditional)": "zh-TW","corsican": "co","croatian": "hr","czech": "cs","danish": "da","dhivehi": "dv","dogri": "doi","dutch": "nl","english": "en","esperanto": "eo","estonian": "et","ewe": "ee","filipino": "tl","finnish": "fi","french": "fr","frisian": "fy","galician": "gl","georgian": "ka","german": "de","greek": "el","guarani": "gn","gujarati": "gu","haitian creole": "ht","hausa": "ha","hawaiian": "haw","hebrew": "iw","hindi": "hi","hmong": "hmn","hungarian": "hu","icelandic": "is","igbo": "ig","ilocano": "ilo","indonesian": "id","irish": "ga","italian": "it","japanese": "ja","javanese": "jw","kannada": "kn","kazakh": "kk","khmer": "km","kinyarwanda": "rw","konkani": "gom","korean": "ko","krio": "kri","kurdish (kurmanji)": "ku","kurdish (sorani)": "ckb","kyrgyz": "ky","lao": "lo","latin": "la","latvian": "lv","lingala": "ln","lithuanian": "lt","luganda": "lg","luxembourgish": "lb","macedonian": "mk","maithili": "mai","malagasy": "mg","malay": "ms","malayalam": "ml","maltese": "mt","maori": "mi","marathi": "mr","meiteilon (manipuri)": "mni-Mtei","mizo": "lus","mongolian": "mn","myanmar": "my","nepali": "ne","norwegian": "no","odia (oriya)": "or","oromo": "om","pashto": "ps","persian": "fa","polish": "pl","portuguese": "pt","punjabi": "pa","quechua": "qu","romanian": "ro","russian": "ru","samoan": "sm","sanskrit": "sa","scots gaelic": "gd","sepedi": "nso","serbian": "sr","sesotho": "st","shona": "sn","sindhi": "sd","sinhala": "si","slovak": "sk","slovenian": "sl","somali": "so","spanish": "es","sundanese": "su","swahili": "sw","swedish": "sv","tajik": "tg","tamil": "ta","tatar": "tt","telugu": "te","thai": "th","tigrinya": "ti","tsonga": "ts","turkish": "tr","turkmen": "tk","twi": "ak","ukrainian": "uk","urdu": "ur","uyghur": "ug","uzbek": "uz","vietnamese": "vi","welsh": "cy","xhosa": "xh","yiddish": "yi","yoruba": "yo","zulu": "zu"}
language_codes2 = {"Afrikaans":"af","Albanian":"sq","Amharic":"am","Arabic":"ar","Armenian":"hy","Azerbaijani":"az","Bengali":"bn","Bosnian":"bs","Bulgarian":"bg","Catalan":"ca","Chinese (Simplified)":"zh","Chinese (Traditional)":"zh-TW","Croatian":"hr","Czech":"cs","Danish":"da","Dari":"fa-AF","Dutch":"nl","English":"en","Estonian":"et","Farsi (Persian)":"fa","Filipino, Tagalog":"tl","Finnish":"fi","French":"fr","French (Canada)":"fr-CA","Georgian":"ka","German":"de","Greek":"el","Gujarati":"gu","Haitian Creole":"ht","Hausa":"ha","Hebrew":"he","Hindi":"hi","Hungarian":"hu","Icelandic":"is","Indonesian":"id","Irish":"ga","Italian":"it","Japanese":"ja","Kannada":"kn","Kazakh":"kk","Korean":"ko","Latvian":"lv","Lithuanian":"lt","Macedonian":"mk","Malay":"ms","Malayalam":"ml","Maltese":"mt","Marathi":"mr","Mongolian":"mn","Norwegian (Bokmål)":"no","Pashto":"ps","Polish":"pl","Portuguese (Brazil)":"pt","Portuguese (Portugal)":"pt-PT","Punjabi":"pa","Romanian":"ro","Russian":"ru","Serbian":"sr","Sinhala":"si","Slovak":"sk","Slovenian":"sl","Somali":"so","Spanish":"es","Spanish (Mexico)":"es-MX","Swahili":"sw","Swedish":"sv","Tamil":"ta","Telugu":"te","Thai":"th","Turkish":"tr","Ukrainian":"uk","Urdu":"ur","Uzbek":"uz","Vietnamese":"vi","Welsh":"cy"}
language_codes3 = {"Afrikaans":"af","Albanian":"sq","Amharic":"am","Arabic":"ar","Armenian":"hy","Assamese":"as","Azerbaijani":"az","Bangla":"bn","Bashkir":"ba","Basque":"eu","Bhojpuri":"bho","Bodo":"brx","Bosnian":"bs","Bulgarian":"bg","Cantonese (Traditional)":"yue","Catalan":"ca","Chinese (Literary)":"lzh","Chinese Simplified":"zh-Hans","Chinese Traditional":"zh-Hant","chiShona":"sn","Croatian":"hr","Czech":"cs","Danish":"da","Dari":"prs","Divehi":"dv","Dogri":"doi","Dutch":"nl","English":"en","Estonian":"et","Faroese":"fo","Fijian":"fj","Filipino":"fil","Finnish":"fi","French":"fr","French (Canada)":"fr-ca","Galician":"gl","Georgian":"ka","German":"de","Greek":"el","Gujarati":"gu","Haitian Creole":"ht","Hausa":"ha","Hebrew":"he","Hindi":"hi","Hmong Daw":"mww","Hungarian":"hu","Icelandic":"is","Igbo":"ig","Indonesian":"id","Inuinnaqtun":"ikt","Inuktitut":"iu","Inuktitut (Latin)":"iu-Latn","Irish":"ga","Italian":"it","Japanese":"ja","Kannada":"kn","Kashmiri":"ks","Kazakh":"kk","Khmer":"km","Kinyarwanda":"rw","Klingon":"tlh-Latn","Klingon (plqaD)":"tlh-Piqd","Konkani":"gom","Korean":"ko","Kurdish (Central)":"ku","Kurdish (Northern)":"kmr","Kyrgyz (Cyrillic)":"ky","Lao":"lo","Latvian":"lv","Lithuanian":"lt","Lingala":"ln","Lower Sorbian":"dsb","Luganda":"lug","Macedonian":"mk","Maithili":"mai","Malagasy":"mg","Malay":"ms","Malayalam":"ml","Maltese":"mt","Maori":"mi","Marathi":"mr","Mongolian (Cyrillic)":"mn-Cyrl","Mongolian (Traditional)":"mn-Mong","Myanmar":"my","Nepali":"ne","Norwegian":"nb","Nyanja":"nya","Odia":"or","Pashto":"ps","Persian":"fa","Polish":"pl","Portuguese (Brazil)":"pt","Portuguese (Portugal)":"pt-pt","Punjabi":"pa","Queretaro Otomi":"otq","Romanian":"ro","Rundi":"run","Russian":"ru","Samoan (Latin)":"sm","Serbian (Cyrillic)":"sr-Cyrl","Serbian (Latin)":"sr-Latn","Sesotho":"st","Sesotho sa Leboa":"nso","Setswana":"tn","Sindhi":"sd","Sinhala":"si","Slovak":"sk","Slovenian":"sl","Somali (Arabic)":"so","Spanish":"es","Swahili (Latin)":"sw","Swedish":"sv","Tahitian":"ty","Tamil":"ta","Tatar":"tt","Telugu":"te","Thai":"th","Tibetan":"bo","Tigrinya":"ti","Tongan":"to","Turkish":"tr","Turkmen":"tk","Ukrainian":"uk","Upper Sorbian":"hsb","Urdu":"ur","Uyghur (Arabic)":"ug","Uzbek":"uz","Vietnamese":"vi","Welsh":"cy","Xhosa":"xh","Yoruba":"yo","Yucatec Maya":"yua","Zulu":"zu"}
language_codes4 = {"Hindi":"hi","Punjabi":"pa","Tamil":"ta","Gujarati":"gu","Kannada":"kn","Bengali":"bn","Marathi":"mr","Telugu":"te","English":"en","Malayalam":"ml","Assamese":"as","Odia":"or","French":"fr","Arabic":"ar","German":"de","Spanish":"es","Japanese":"ja","Italian":"it","Dutch":"nl","Portuguese":"pt","Vietnamese":"vi","Indonesian":"id","Urdu":"ur","Chinese (Simplified)":"zh-CN","Chinese (Traditional)":"zh-TW","Kashmiri":"ksm","Konkani":"gom","Manipuri":"mni-Mtei","Nepali":"ne","Sanskrit":"sa","Sindhi":"sd","Bodo":"bodo","Santhali":"snthl","Maithili":"mai","Dogri":"doi","Malay":"ms","Filipino":"tl"}

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
encryption_key = os.environ['enc_key']
fernet = Fernet(encryption_key)

profanity_tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
profanity_model = BertModel.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
er_loaded = torch.load('./encoded_representation.pt')
with open('./vocab.json', 'r') as json_file:
    swear_words = json.load(json_file)
    swear_words1 = list(swear_words.keys())
er_loaded = {intent: intent_encoding for obj in er_loaded for intent, intent_encoding in obj.items()}

aws_client = boto3.client('translate',region_name='us-east-1',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCCESS_KEY']
)

# Decrypt the encrypted data
with open("encrypted_data.json", "rb") as encrypted_file:
    encrypted_data = encrypted_file.read()
    decrypted_data = fernet.decrypt(encrypted_data)

with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
    temp_file.write(decrypted_data)
    temp_file_path = temp_file.name

# Set the temporary file path as the Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path
vertexai.init(project=os.environ['project_id'], location='us-central1')

perspective_client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)


m,t,ans, sty="","","",""
flag=False

def stream(input_text):
    llm = ChatOpenAI(model='gpt-4')

    conversation = ConversationChain(
        prompt=PromptTemplate(
            input_variables=['history', 'input'],
            template=(
                "Imagine you are an AI Content Generator. Your goal is to create unique, engaging, and high-quality content based on the given input with an example. if language is sepcified your task is to generate content in the language specified by the user while keeping text within quotes in English (like this: ' we will rock you') and the rest in the chosen language. Use the appropriate devanagari script for the chosen language. If the user doesn't specify a language, default to English. Ensure that all sentences are grammatically correct. When handling language specification, make sure that any text within quotes remains in English while the rest of the content is in the chosen language. If the user provides text formatting instructions like lists, be sure to follow them. Also, ensure that the generated content contains no questions, and if a word count is specified, do not exceed it.  \n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
            )),
        llm=llm,
    )
    return conversation.run(input_text)

def predict_gpt(input_text, selected_model, web_search, style, temperature, choice, frequency_penalty, presence_penalty, top_p):
    selected_model = selected_model.lower()
    if selected_model=='gpt-4-turbo':
        selected_model='gpt-4-1106-preview'
    if web_search:
        llm = ChatOpenAI(
            model_name=selected_model,
            temperature=0.5,
        )
    
        DDGsearch = DuckDuckGoSearchRun()
    
        PREFIX = """
                    Imagine you are an AI Content Generator.
                    Your goal is to create unique, engaging and descriptive content based on the given input with an example. 
                    If a word count is specified in given input, Strictly maintain the word limit.
                    Ensure that all sentences are grammatically correct. 
                    If the user provides text formatting instructions like lists, be sure to follow them. 
                    Avoid any url links in the content.
                    Also, ensure that the generated content contains no questions.
                """
        SUFFIX = """
        Begin!
        Instructions: 
        {input}
        {agent_scratchpad}
        """
    
        tools = [
            Tool(
                name="Duck Duck Go Search Results Tool",
                func=DDGsearch.run,
                description="Useful for search for information on the internet",
            ),
        ]
    
        runAgent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            agent_kwargs={"prefix": PREFIX, "suffix": SUFFIX},
            handle_parsing_errors=True,
        )
    
        context = str(runAgent.run(input_text))
    
        client = OpenAI()
        conversation = [
            {"role": "system", "content": "You are an AI language model."},
            {
                "role": "user",
                "content": f""" Input: {input_text} context: {context}. \n\n\n
                            Strictly Follow the Below Instructions: 
                            1. Imagine you are an Content Generator.
                            2. Follow the specified style: {style}
                            3. Avoid 'Input' in your content, avoid same text of context and questions in content. 
                            4. Your goal is to create unique, engaging and descriptive content on your own for given input with an example having given context as additional knowledge .
                            5. If a word count is specified in given input, Strictly maintain the word limit.
                            6. Ensure that all sentences are grammatically correct. 
                            7. If the user provides text formatting instructions like lists, be sure to follow them. 
                            8. Also, ensure that the generated content contains no questions.
                            9. Omit any references to the AI model.""",
            }
            
        ]
        res = client.chat.completions.create(model=selected_model, messages=conversation, temperature=temperature, n=choice, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, top_p=top_p)
        return ' '.join(['Choice 1:\n\n'+res.choices[i].message.content.replace("\n", "")+'\n\n\n' for i in range(len(res.choices))])
        
    else:
        client = OpenAI()
        conversation = [
            {"role": "system", "content": "You are an AI language model."},
            {
                "role": "user",
                "content": f""" Input: {input_text} context: {context}. \n\n\n
                            Strictly Follow the Below Instructions: 
                            1. Imagine you are an Content Generator.
                            2. Follow the specified style: {style}
                            3. Avoid 'Input' in your content, avoid same text of context and questions in content. 
                            4. Your goal is to create unique, engaging and descriptive content on your own for given input with an example having given context as additional knowledge .
                            5. If a word count is specified in given input, Strictly maintain the word limit.
                            6. Ensure that all sentences are grammatically correct. 
                            7. If the user provides text formatting instructions like lists, be sure to follow them. 
                            8. Also, ensure that the generated content contains no questions.
                            9. Omit any references to the AI model.""",
            }
        ]
        res = client.chat.completions.create(model=selected_model, messages=conversation, temperature=temperature, n=choice, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, top_p=top_p)
        return ' '.join(['Choice 1:\n\n'+res.choices[i].message.content.replace("\n", "")+'\n\n\n' for i in range(len(res.choices))])
        
        

def predict_palm2(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    style: str = ''
    ) :
        
    infoString = [
        {"role": "system", "content": "You are an AI content Generator. Omit any references to the AI model."},
        {
            "role": "user",
            "content": "Your goal is to create unique, engaging and descriptive content on :"+ content,
        },
        {
            "role": "user",
            "content": f"Incorporate the specified style : {style} in generating content",
        }
    ]

    content = "follow the instructions in this conversation structure below\n" + str(
        infoString
    )
        
    model = TextGenerationModel.from_pretrained(model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,)
    return response.text

url = "https://api.perplexity.ai/chat/completions"

def predict_llama2(text, style, temperature, frequency_penalty, top_p, top_k):
  payload = {
      "model": "llama-2-70b-chat",
      "messages": [
          {
              "role": "system",
              "content": "Imagine you are an AI Content Generator. strictly Avoid adding introductory sentence like 'Sure, here is some' "
          },
          {
              "role":"system",
              "content": "Ensure that all sentences are grammatically correct. If the user provides text formatting instructions like lists, be sure to follow them. "
          },
          {
              "role": "system",
              "content": "Also, ensure that the generated content contains no questions. Avoid greetings and conclusions."
          },
          {
              "role": "system",
              "content": "Strictly Avoid Incomplete Sentences. Try to complete the sentence."
          },
          {
              "role": "user",
              "content": text+f" Follow the specified style: {style}"
          }
      ],
      "temperature":temperature,
      "frequency_penalty":frequency_penalty,
      "top_p":top_p,
      "top_k":top_k
  }
  headers = {
      "accept": "application/json",
      "content-type": "application/json",
      "authorization": os.environ['pplx_api_key']
  }

  response = requests.post(url, json=payload, headers=headers)

  return json.loads(response.text)['choices'][0]['message']['content']


def split_text(input_text, max_chunk_length=4995):
    sentences = sent_tokenize(input_text)

    # Initialize variables
    chunks = []
    current_chunk = ""

    # Iterate through sentences
    for sentence in sentences:
        # Check if adding the sentence to the current chunk exceeds the max length
        if len(current_chunk) + len(sentence) <= max_chunk_length:
            current_chunk += sentence + " "
        else:
            # Save the current chunk and start a new one with the current sentence
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def gtranslate(language, text):
    return GoogleTranslator(source="auto", target=language_codes1[language]).translate(text)

def google_translation(language, text):
    if len(text)>=5000:
        print('Character length greater than 5000')
        result_chunks = split_text(text)
        language = [language]*len(result_chunks)
        with concurrent.futures.ThreadPoolExecutor() as executor:
          result_chunks = list(executor.map(gtranslate, language, result_chunks))
        return ' '.join(result_chunks)
    return GoogleTranslator(
                        source="auto", target=language_codes1[language]
                    ).translate(text)

def aws_translate(language,text):
    response = aws_client.translate_text(
    Text=text,
    SourceLanguageCode='auto',
    TargetLanguageCode=language_codes2[language],
    )
    return response['TranslatedText']

def azure_translate(language,text):
    key = os.environ['azure_api_key']
    endpoint = "https://api.cognitive.microsofttranslator.com"
    location = "eastus"
    
    path = '/translate'
    constructed_url = endpoint + path
    
    params = {
        'api-version': '3.0',
        'to': [language_codes3[language]]
    }
    
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    
    # You can pass more than one object in body.
    body = [{'text': text}]
    
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    return response[0]["translations"][0]['text']

def devnagri_translate(language, text):
    url = "https://app.devnagri.com/api/translate-sentence-api"
    result = detect(text=text, low_memory=False)
    payload = {
        "key": os.environ['devnagri_api_key'],
        "sentence": text,
        "src_lang": result['lang'],
        "dest_lang": language_codes4[language],
    }

    response = requests.post(url, data=payload)

    if response.status_code == 200:
        return json.loads(response.text)["translated_text"]
    else:
        return "Couldn't provide response for the desired language."    

def translate(language_provider, language, text):
    if language_provider=='Google Translate':
        return google_translation(language, text)
    elif language_provider == 'AWS Translate':
        return aws_translate(language, text)
    elif language_provider == 'Azure AI Translator':
        return azure_translate(language, text)
    else:
        return devnagri_translate(language, text)

def postprocess(text):
    return text.replace("\n\n", " ").replace("\r", "").replace("\n", " ")

def openai_profanity(text):
    client = OpenAI()
    res = client.moderations.create(input=text)
    return json.loads(res.json())['results'][0]['flagged']

def google_profanity(text):
    analyze_request = {
  'comment': { 'text': text },
  'requestedAttributes': {'PROFANITY': {}, 'TOXICITY': {}, 'INSULT': {}, 'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK':{}, 'THREAT': {}, 'OBSCENE': {}, 'SEXUALLY_EXPLICIT':{}, 'FLIRTATION':{}}
    }

    response = perspective_client.comments().analyze(body=analyze_request).execute()
    for attribute, scores in response.get('attributeScores', {}).items():
        summary_score = scores.get('summaryScore', {}).get('value', 0)
        if summary_score > 0.6:
            print(f"{attribute}: {summary_score}")
    return any([scores['summaryScore']['value'] > 0.6 for scores in response.get('attributeScores', {}).values()])

def compare(sentences):

    inputs = profanity_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        embeddings = profanity_model(**inputs).last_hidden_state

    matching_intents = []

    for intent, intent_encoding in er_loaded.items():
        # Calculate similarities for the entire batch
        similarities = cosine_similarity(embeddings.mean(dim=1), intent_encoding.numpy())
            
        # Check if any similarities in the batch exceed the threshold
        for i, sim in enumerate(similarities):
            if sim > 0.85:
                matching_intents.append((sentences[i], intent)) 
                return matching_intents
    return []

def gida_profanity(text):
    text = text.lower()

    if list(filter(lambda value: value in swear_words1, text.split())):
        return True

    x = [(word, W) for word in swear_words1 for W in text.split() if word in W]

    for X in x:
        if len(X[0]) / (len(X[1]) - len(X[0])) > 0.65:
            return True

    # Split text into sentences
    sentences = re.split(r'[.?!]\s', text)

    batch_size = 5

    # Initialize executor
    executor = concurrent.futures.ThreadPoolExecutor()

    # Process sentences in batches
    results = []
    futures = []

    def process_batch(sentences):
        return compare(sentences)

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        # Submit the batch processing to the executor
        future = executor.submit(process_batch, batch)
        futures.append(future)

    # Wait for all futures to complete and gather the results
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        results.extend(result)

    executor.shutdown()

    # Check if any batch had matching intents
    if any(results):
        return True

    return False

def profanity(profanity_provider, text):
    if profanity_provider == 'OpenAI':
        return openai_profanity(text)
    elif profanity_provider == 'Google':
        return google_profanity(text)
    else:
        return gida_profanity(text)
        

# Define a function to generate content based on the selected model
def generate_content(selected_model, web_search, text, style, temperature, choice, frequency_penalty, presence_penalty, top_p, top_k, profanity_provider, language_provider, language=None):
    global m,t,ans
    selected_model=selected_model.split(" ")[0]
    print(selected_model)
    print(web_search)
    print(text)
    print(style)
    print(temperature)
    print(choice)
    print(frequency_penalty)
    print(presence_penalty)
    print(top_p)
    print(top_k)
    print(profanity_provider)
    print(language_provider)
    print(language)
    if profanity(profanity_provider, text+' '+style):
        yield "Profanity has been detected in your text. Please refrain from using inappropriate language."
        
    if selected_model==m and text==t and flag==web_search and style == sty:
        ans = translate(language_provider, language, ans)
        print(ans)
        yield ans
    else:
        m=selected_model
        t=text
        flag=web_search
        sty=style
        try:
            if selected_model[:3] == 'GPT':
              ans = predict_gpt(text, selected_model, web_search, style, temperature, choice, frequency_penalty, presence_penalty, top_p)
              if not language:
                yield ans
              else:
                ans = translate(language_provider, language, ans)
                yield ans
        
            elif selected_model == 'Llama2':
              ans = predict_llama2(text, style, temperature, frequency_penalty, top_p, top_k )
              if not language:
                yield ans
              else:
                ans = translate(language_provider, language, postprocess(ans))
                yield ans
        
            elif selected_model == 'PaLM2':
              ans = predict_palm2(os.environ['project_id'], "text-bison", temperature, 512, top_p, top_k, f'Avoid input in your response. Imagine you are an AI Content Generator. Try to generate content for {text} in specified style {style}.  if the word "article" is specified in input, then you try to add title to it otherwise strictly avoid adding title. If the user provides text formatting instructions like lists, be sure to follow them. Also, ensure that the generated content contains no questions, and if a word count is specified, do not exceed it. Your goal is to create unique, engaging, and high-quality content based on the given input.', "us-central1")
              if not language:
                yield ans.replace("**", "").replace("*", "")
              else:
                ans = ans = translate(language_provider, language, postprocess(ans)).replace("**", "").replace("*", "")
                yield ans
        except:
            pass
        finally:
            print(ans)


# Dropdown choices with model name and pricing
model_dropdown_choices = [
    "GPT-3.5-turbo ($0.0002 + $0.0002 / 100 output words)",
    "GPT-4-turbo ($0.002 + $0.004 / 100 output words)",
    "GPT-4 ($0.006 + $0.008 / 100 output words)",
    "PALM2 ($0.0001 + $0.00006 / 100 output words)",
    "Llama2 ($0.001 / requests)"
]

# Language choices based on language provider
languages = ['Google Translate', 'AWS Translate', 'Azure AI Translator', 'Devnagri.com']
language_choices_map = {
    "Google Translate": ['afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'assamese', 'aymara', 'azerbaijani', 'bambara', 'basque', 'belarusian', 'bengali', 'bhojpuri', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dhivehi', 'dogri', 'dutch', 'english', 'esperanto', 'estonian', 'ewe', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek', 'guarani', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hindi', 'hmong', 'hungarian', 'icelandic', 'igbo', 'ilocano', 'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'kinyarwanda', 'konkani', 'korean', 'krio', 'kurdish (kurmanji)', 'kurdish (sorani)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lingala', 'lithuanian', 'luganda', 'luxembourgish', 'macedonian', 'maithili', 'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'meiteilon (manipuri)', 'mizo', 'mongolian', 'myanmar', 'nepali', 'norwegian', 'odia (oriya)', 'oromo', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'quechua', 'romanian', 'russian', 'samoan', 'sanskrit', 'scots gaelic', 'sepedi', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'tatar', 'telugu', 'thai', 'tigrinya', 'tsonga', 'turkish', 'turkmen', 'twi', 'ukrainian', 'urdu', 'uyghur', 'uzbek', 'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu'],
    "AWS Translate": ['Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Azerbaijani', 'Bengali', 'Bosnian', 'Bulgarian', 'Catalan', 'Chinese (Simplified)', 'Chinese (Traditional)', 'Croatian', 'Czech', 'Danish', 'Dari', 'Dutch', 'English', 'Estonian', 'Farsi (Persian)', 'Filipino, Tagalog', 'Finnish', 'French', 'French (Canada)', 'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Kannada', 'Kazakh', 'Korean', 'Latvian', 'Lithuanian', 'Macedonian', 'Malay', 'Malayalam', 'Maltese', 'Marathi', 'Mongolian', 'Norwegian (Bokmål)', 'Pashto', 'Polish', 'Portuguese (Brazil)', 'Portuguese (Portugal)', 'Punjabi', 'Romanian', 'Russian', 'Serbian', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Spanish (Mexico)', 'Swahili', 'Swedish', 'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese', 'Welsh'],
    "Azure AI Translator": ['Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Azerbaijani', 'Bangla', 'Bashkir', 'Basque', 'Bhojpuri', 'Bodo', 'Bosnian', 'Bulgarian', 'Cantonese (Traditional)', 'Catalan', 'Chinese (Literary)', 'Chinese Simplified', 'Chinese Traditional', 'chiShona', 'Croatian', 'Czech', 'Danish', 'Dari', 'Divehi', 'Dogri', 'Dutch', 'English', 'Estonian', 'Faroese', 'Fijian', 'Filipino', 'Finnish', 'French', 'French (Canada)', 'Galician', 'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hebrew', 'Hindi', 'Hmong Daw', 'Hungarian', 'Icelandic', 'Igbo', 'Indonesian', 'Inuinnaqtun', 'Inuktitut', 'Inuktitut (Latin)', 'Irish', 'Italian', 'Japanese', 'Kannada', 'Kashmiri', 'Kazakh', 'Khmer', 'Kinyarwanda', 'Klingon', 'Klingon (plqaD)', 'Konkani', 'Korean', 'Kurdish (Central)', 'Kurdish (Northern)', 'Kyrgyz (Cyrillic)', 'Lao', 'Latvian', 'Lithuanian', 'Lingala', 'Lower Sorbian', 'Luganda', 'Macedonian', 'Maithili', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian (Cyrillic)', 'Mongolian (Traditional)', 'Myanmar', 'Nepali', 'Norwegian', 'Nyanja', 'Odia', 'Pashto', 'Persian', 'Polish', 'Portuguese (Brazil)', 'Portuguese (Portugal)', 'Punjabi', 'Queretaro Otomi', 'Romanian', 'Rundi', 'Russian', 'Samoan (Latin)', 'Serbian (Cyrillic)', 'Serbian (Latin)', 'Sesotho', 'Sesotho sa Leboa', 'Setswana', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali (Arabic)', 'Spanish', 'Swahili (Latin)', 'Swedish', 'Tahitian', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tibetan', 'Tigrinya', 'Tongan', 'Turkish', 'Turkmen', 'Ukrainian', 'Upper Sorbian', 'Urdu', 'Uyghur (Arabic)', 'Uzbek', 'Vietnamese', 'Welsh', 'Xhosa', 'Yoruba', 'Yucatec Maya', 'Zulu'],
    "Devnagri.com": ['Hindi', 'Punjabi', 'Tamil', 'Gujarati', 'Kannada', 'Bengali', 'Marathi', 'Telugu', 'English', 'Malayalam', 'Assamese', 'Odia', 'French', 'Arabic', 'German', 'Spanish', 'Japanese', 'Italian', 'Dutch', 'Portuguese', 'Vietnamese', 'Indonesian', 'Urdu', 'Chinese (Simplified)', 'Chinese (Traditional)', 'Kashmiri', 'Konkani', 'Manipuri', 'Nepali', 'Sanskrit', 'Sindhi', 'Bodo', 'Santhali', 'Maithili', 'Dogri', 'Malay', 'Filipino']
}

def rs_change(rs):
    return gr.update(choices=language_choices_map[rs], value=None)

# Create a Gradio interface
with gr.Blocks() as app:
    model_info = gr.Dropdown(label="Select Text Generation Model", choices=model_dropdown_choices)
    web_search_checkbox = gr.Checkbox(label="Enable Web Search", value=False, info='Available only for GPT Models', show_label=True)
    text_input = gr.Textbox(label="Input Text")
    style = gr.Textbox(label="Enter the style you wanted the generated to be in")
    temperature = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, label='Select the temperature',randomize=True ,show_label=True, info="Controls the randomness of the generated text.")
    choice = gr.Slider(minimum=1, maximum=5, step=1, label='Select the number of choice' ,value=1, show_label=True, info="Only for GPT models")
    frequency_penalty = gr.Slider(minimum=-2.0, maximum=2.0, step=0.1, label='Select the frequency penalty', randomize=True ,show_label=True, info="Only for GPT Models/LLama2. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.")
    presence_penalty = gr.Slider(minimum=-2.0, maximum=2.0, step=0.1, label='Select the presence penalty', randomize=True ,show_label=True, info="Only for GPT Models. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics")
    top_p = gr.Slider(minimum=0, maximum=1, step=0.1, label='Select the top_p value', show_label=True, info="Generally recommend altering this or temperature but not both.")
    top_k = gr.Slider(minimum=0, maximum=1024, step=1, label='Select the top_k value', show_label=True,info="Only for Llama2 and PALM2. The number of tokens to keep for highest top-k filtering.")
    profanity_provider = gr.Dropdown(label="Profanity Detection Provider", choices=['Google', 'OpenAI', 'Gida'], value='Google')
    language_provider = gr.Dropdown(label="Language Translation Provider", choices=languages, value='Google Translate')
    language_dropdown = gr.Dropdown(label="Language", choices=language_choices_map['Google Translate'], interactive=True)

    language_provider.change(fn=rs_change,  inputs=[language_provider], outputs=[language_dropdown])
    outputs = gr.Textbox(label="Generated Content")

    gr.Interface(
        fn=generate_content,
        inputs=[model_info, web_search_checkbox, text_input, style, temperature, choice, frequency_penalty, presence_penalty, top_p, top_k, profanity_provider, language_provider ,language_dropdown ],
        outputs=outputs,
        title="AI Content Generator",
        description="Your Ultimate AI Content Wizard! You can also specify the language using a dedicated dropdown field."
    )

app.launch(debug=True, share=True)