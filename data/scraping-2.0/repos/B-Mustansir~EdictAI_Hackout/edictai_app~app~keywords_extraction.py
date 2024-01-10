from keybert import KeyBERT
import openai 
# from .env import *

def keywords_extraction(sentence):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(sentence,keyphrase_ngram_range=(1, 2),top_n=1)
    return(keywords[0][0])

# def keywords_extraction(sentence):
#     openai.organization = generate_script_openai_organization
#     openai.api_key = generate_script_openai_api_key
#     # print(openai.Model.list())

#     completion = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "I want you to act as a Keyword extractor. I will provide you with a sentence from news article and you will generate keywords from that sentence to search image based on these keywords"},
#         {"role": "user", "content": f'''
#         Given the following sentence, please provide a concise query (1-2 words) that can be used to search for relevant images on Google.
#         Sentence: {sentence}
#         Note: Output the query only dont write anything else. Keep the query as simple as possible so that image could be searched easily. 
#         '''}
#     ]
#     )
#     print(completion.choices[0].message.content)
#     return (completion.choices[0].message.content)