from keybert import KeyBERT

kw_model = KeyBERT()
keywords = kw_model.extract_keywords("If someone were to ask me- if there is one place you must visit in Uttarakhand which place would it be, I would say you must visit Parvati Kund and Jageshwar Temples in the Kumaon region of the state. The natural beauty and divinity will leave you spellbound. ",keyphrase_ngram_range=(1, 1)) 
only_keywords = []
if len(keywords)>3:
    keywords = keywords[0:3]
for key in keywords: 
    only_keywords.append(key[0])
print(only_keywords)

# import openai
# from keybert.llm import OpenAI
# from keybert import KeyLLM
# from sentence_transformers import SentenceTransformer

# # Extract embeddings
# # model = SentenceTransformer('all-MiniLM-L6-v2')
# # embeddings = model.encode(MY_DOCUMENTS, convert_to_tensor=True)

# openai.api_key = "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"
# sentence = "The Prime Minister, Shri Narendra Modi congratulated Raunak Sadhwani on the remarkable victory at the FIDE World Junior Rapid Chess Championship 2023."

# # llm = OpenAI()
# # kw_model = KeyLLM(llm)
# # keywords = kw_model.extract_keywords(sentence) 
# # print(keywords)

# # Extract keywords
# # keywords = kw_model.extract_keywords(MY_DOCUMENTS, embeddings=embeddings, threshold=.75)

# import openai
# from keybert.llm import OpenAI
# from keybert import KeyLLM
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(sentence, convert_to_tensor=True)
# llm = OpenAI()
# kw_model = KeyLLM(llm)
# keywords = kw_model.extract_keywords(sentence, embeddings=embeddings)
# print(keywords)

import os
import openai
openai.organization = ""
openai.api_key = ""

sentence = '''The Prime Minister, Shri Narendra Modi congratulated Raunak Sadhwani on the remarkable victory at the FIDE World Junior Rapid Chess Championship 2023.'''

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0.5,
    messages=[
        {
            "role": "system",
            "content": '''
            You will be provided with a block of text, and your task is to extract a list of keywords from it.
            Note: Keywords extracted would be used as a query to search for images on search engines. 
            Please avoid unnecessary details or tangential points.
            '''
        },
        {
            "role": "user",
            "content": sentence
        }
    ]
)
print (response['choices'][0]['message']['content'])