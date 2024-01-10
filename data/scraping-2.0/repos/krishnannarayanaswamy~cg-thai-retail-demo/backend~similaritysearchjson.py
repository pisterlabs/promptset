from flask import Flask, request
from langdetect import detect_langs
from flask_cors import CORS
import openai
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import dict_factory
from cassandra.query import SimpleStatement
import pandas as pd 
import os
import json
import re

cass_user = os.environ.get('cass_user')
cass_pw = os.environ.get('cass_pw')
scb_path =os.environ.get('scb_path')
open_api_key= os.environ.get('openai_api_key')
keyspace = os.environ.get('keyspace')
table_name = os.environ.get('table')
model_id = "text-embedding-ada-002"
#model_id='embed-multilingual-v2.0'
openai.api_key = open_api_key
cloud_config= {
  'secure_connect_bundle': scb_path
}

auth_provider = PlainTextAuthProvider(cass_user, cass_pw)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()
session.set_keyspace(keyspace)

app = Flask(__name__)
CORS(app)

@app.route('/getbrand', methods=['GET'])
def brand_autofill():
    query = SimpleStatement(
    f"""
    SELECT count(product_id), brand
    FROM {keyspace}.brand_cg
    GROUP BY brand"""
    )

    results = session.execute(query)
    top_brands = results._current_rows
    print(len(top_brands))

    hintArray = []
    if len(top_brands) != 0:
        branddf = pd.DataFrame(top_brands)
        for index, row in branddf.iterrows():
            pattern = row['brand']
            pattern = re.sub(r'[()]', '', pattern)
            pattern = pattern.replace("-", "")
            pattern = pattern.replace(".", "")
            pattern = pattern.replace(",", "")
            pattern = pattern.replace(" ", "")
            hintArray.append(pattern.lower()) 
    print(hintArray)
    resvalues = dict()
    resvalues['brands'] = hintArray
    return resvalues

def detect_brand(customer_query):
    message_objects = []
    message_objects.append({"role":"system",
                            "content":"Extract product brand name, category, keywords, price range from a user query and respond the detected pair in a json formatted string like this .. {\"brand\": \"brand\", \"category\":\"\", \"keywords\":\"\", \"price\": { \"amount\": 1.0, \"operator\":\"\"}} operator can be \">\" , \"<\". Respond only json object, no need any description or code. if brand is not found, leave it empty"})

    message_objects.append({"role":"user",
                            "content": customer_query})

    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=message_objects
    )

    brand_category = completion.choices[0].message['content']

    filter_keyword=json.loads(brand_category)
    brand = str(filter_keyword['brand'].upper())
    print("System detected brand with GPT 4:" + brand)

def translate_lang(query):
    message_objects = []
    message_objects.append({"role":"user",
                        "content": "Translate these Thai words to English:'" +  query + "'"})
    completion = openai.ChatCompletion.create(
    model="gpt-4", 
    messages=message_objects
    )
    text_in_en = completion.choices[0].message.content
    print("System translated text with GPT 4:" + text_in_en)
   
    return text_in_en


@app.route('/getkeywords',methods=["POST"])
def get_keywords():
    products = request.json.get('products')
    language = request.json.get('language')
    print(language)

    thailanguage = ''
    if language == "th":
        thailanguage = "in Thai language and"

    message_objects = []
    message_objects.append({"role":"system",
                            "content":"You are a eCommerce home improvement recommendation assistant helping users choose products"})
    #if language == "th":
    #    message_objects.append({"role":"system",
    #                           "content":"Respond in Thai Language"})
    message_objects.append({"role":"user",
                            "content": f"I need exactly one unique, exclusive keyword for every product in the list. Return the results {thailanguage} in a JSON array format [""{'keyword1':'product name'},{'keyword2':'product name'},{'keyword3':'product name'},{'keyword4':'product name'},{'keyword5':'product name'}]. Each keyword should represent a feature or a function, for eg, Price, Quality, etc. Response should have one uniquely identifiying keyword for each product. Price should always be on the keyword list"})
    products_list = []
    for row in products:          
        brand_dict = {'role': "user", "content": f"Name: {row['productname']}, Description: {row['shortdescription']}, Brand: {row['name']}, Price: {row['price']}"}
        products_list.append(brand_dict)
    message_objects.extend(products_list)    
    print(message_objects)
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message_objects,
        temperature=0.2,
    )
    keywords_list = completion.choices[0].message['content']    
    print(keywords_list)
    valid_json_string = keywords_list.replace("'", '"')
    print(valid_json_string) 
    values = dict()
    values['keywords'] = json.loads(valid_json_string)
    values['language'] = language
    return values

@app.route('/selectitems',methods=["POST"])
def select_items():
    #customer_query = request.json.get('newQuestion')
    #langs = detect_langs(customer_query)
    language = request.json.get('language')
    products = request.json.get('products')
    kw = request.json.get('keyword')
    index = request.json.get('index')
    thailanguage = ''
    if language == "th":
        thailanguage = "Respond in Thai Language."
    print(language)
    keyword = ''    
    for key in kw:
        keyword = key    
    message_objects = []
    message_objects.append({"role":"system",
                            "content":f"You are a eCommerce home improvement recommendation assistant helping users choose products. {thailanguage} Clearly explain Why you are making a recommendation in a separate 'Why' section"})
    #if language == "th":
    #    message_objects.append({"role":"system",
    #                           "content":"Respond in Thai Language"})
   # message_objects.append({"role":"system",
   #                         "content":"Clearly explain Why you are making a recommendation in a separate 'Why' section"})
    message_objects.append({"role":"user",
                            "content":f"Choose one product exclusively based on the preference: {keyword}. Clearly explain Why this product is better suited for the given preference"})
    products_list = []
    row = products[index]    
    brand_dict = {'role': "user", "content": f"Name: {row['productname']}, Description: {row['shortdescription']} {row['longdescription']}, Brand: {row['name']}, Price: {row['price']}"}
    products_list.append(brand_dict)
    message_objects.extend(products_list)    
    print(message_objects)
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message_objects,
        temperature = 0.2
    )    
    human_readable_response = completion.choices[0].message['content']
    print(human_readable_response)
    values = dict()        
    values['botresponse'] = human_readable_response
    return values

@app.route('/similaritems', methods=['POST'])
def ann_similarity_search():
    #customer_query='สีที่ดีที่ละลายได้อย่างสวยงาม'
    customer_query = request.json.get('newQuestion')
    brand = request.json.get('text').upper()
    print(brand)
    #english_customer_text= translator.translate(customer_query)
    #print(english_customer_text.text)
    langs = detect_langs(customer_query)
    language = langs[0].lang
    print(language)
    
    customer_text = []
    if language == "th":
        customer_query_en = translate_lang(customer_query)
    else:
        customer_query_en = customer_query

    customer_text.append(customer_query)
    
    if (brand == "" or brand is None):
       detect_brand(customer_query_en)
    #brand = ""

    embeddings = openai.Embedding.create(input=customer_query_en, model=model_id)['data'][0]['embedding']
   # embeddings = response.embeddings[0]
    column = "openai_description_embedding_en"
    query = SimpleStatement(
        f"""
        SELECT product_id, brand,image_link, saleprice,product_categories, product_name, product_name_en, short_description, short_description_en, long_description, long_description_en
        FROM {keyspace}.products_cg_hybrid
        ORDER BY {column} ANN OF {embeddings}
        LIMIT 10 """
        )

    if brand != "" and brand != "null" and brand != "None" and brand != "Unknown" and brand != "N/A" and brand != "Not specified":
        query = SimpleStatement(
            f"""
            SELECT product_id,brand,image_link,saleprice,product_categories, product_name, product_name_en, short_description, short_description_en, long_description, long_description_en
            FROM {keyspace}.products_cg_hybrid
            WHERE product_name : ' + {brand} + '
            ORDER BY {column} ANN OF {embeddings}
            LIMIT 10 """
            )
  
    #print(query)
    results = session.execute(query)
    top_products = results._current_rows
    print(len(top_products))

    if len(top_products) == 0:
        query = SimpleStatement(
                f"""
                SELECT product_id,brand,image_link,saleprice,product_categories, product_name, product_name_en, short_description, short_description_en, long_description, long_description_en
                FROM {keyspace}.products_cg_hybrid
                ORDER BY {column} ANN OF {embeddings}
                LIMIT 10 """
                )

        results = session.execute(query)
        top_products = results._current_rows

    response = []
    for r in top_products:
        image = r.image_link.split(".jpg", 1)
        image = image[0] + ".jpg"
        if language == "th":
            response.append({
                'id': r.product_id,
                'name': r.brand,
                'productname': r.product_name,
                'shortdescription': r.short_description,
                'longdescription': r.long_description,
                'price': r.saleprice,
                'category': r.product_categories,
                'image_link': image
            })
        else:
            response.append({
                'id': r.product_id,
                'name': r.brand,
                'productname': r.product_name_en,
                'shortdescription': r.short_description_en,
                'longdescription': r.long_description_en,
                'price': r.saleprice,
                'category': r.product_categories,
                'image_link': image
            })
    #print(response)

    #message_objects_gpt = []
    #message_objects_gpt.append({"role":"system",
    #                        "content":"You are a friendly, conversational home improvement retail shopping assistant.Use the following context including product names, descriptions, and keywords to show the shopper whats available. Prices are in USD"})
    
    #message_objects_gpt.append({"role":"user",
    #                        "content":"Provide answers in the language of the query"})
    
    #message_objects_gpt.append({"role":"user",
    #                        "content": customer_query})

    #message_objects_gpt.append({"role":"user",
    #                        "content": "Please give me a detailed explanation based on your recommendations provided earlier"})

    #message_objects_gpt.append({"role":"user",
    #                       "content": "Please be friendly and talk to me like a person, don't just give me a list of recommendations"})

    #message_objects_gpt.append({"role": "assistant",
   #                         "content": "I found these products I would recommend"})

    #products_list = []

    #for row in response:  
    #    print(row)          
    #    brand_dict = {'role': "assistant", "content": f"{row['productname']}, {row['shortdescription']}, {row['name']}, {row['price']}"}
   #     products_list.append(brand_dict)

   # message_objects_gpt.extend(products_list)
   # message_objects_gpt.append({"role": "assistant", "content":"Here's my summarized recommendation of products, and why it would suit you:"})

    #completion = openai.ChatCompletion.create(
    #   model="gpt-4",
    #   messages=message_objects_gpt
   # )
    #print(completion)
    #human_readable_response = completion.choices[0].message['content']
    #print(human_readable_response) 

    values = dict()
    values['products'] = response
    values['language'] = language
    #values['botresponse'] = human_readable_response

    return values

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)