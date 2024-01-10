from openai import OpenAI
import numpy as np
from src.utils.managePaths import mp
from src.utils.credentials import api_key_openai
import json

jsonCollectionsPath=mp.get_path_collections_embeddings()
jsonProductsPath=mp.get_path_product_embeddings()
def get_openai_embedding(text:str):
    client = OpenAI(api_key=api_key_openai)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding
def get_openai_embeddings(texts:list):
    client = OpenAI(api_key=api_key_openai)
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    r=[x.embedding for x in response.data]
    return r
def get_similarity(embed1, embed2):
    similarity = np.dot(embed1, embed2)
    return similarity
def get_data_google_sheet():
    with open("data.json","r",encoding="utf-8") as json_file:
        data=json.load(json_file)
    return data

def search_collection(collection)->dict:
    with open(jsonCollectionsPath,"r",encoding="utf-8") as json_file:
        data=json.load(json_file)
    collecs=data['shopify']['collecciones']
    for collec in collecs:
        if collection in collec.keys():
            return collec

def add_collection(collection)->None:
    with open(jsonCollectionsPath,"r",encoding="utf-8") as json_file:
            data=json.load(json_file)
    collecs=data['shopify']['collecciones']
    collecs.append(collection)

    with open(jsonCollectionsPath,"w",encoding="utf-8") as json_file:
        json.dump(data,json_file,indent=4,ensure_ascii=False)

def get_collection(collection:str)->None:
    r=search_collection(collection)
    if r:
        print("ya existe")
        print(list(r.keys())[0])
    else:
        print("no existe,agrenado...")
        add_collection({collection:get_openai_embedding(collection)})

def search_product_embedding(product:str)->None:
    with open(jsonProductsPath,"r",encoding="utf-8") as json_file:
        data=json.load(json_file)
    products=data['shopify']['productos']
    for pro in products:
        if product['sku']==pro['sku']:
            return pro
    return None

def add_product(product)->None:
    with open(jsonProductsPath,"r",encoding="utf-8") as json_file:
            data=json.load(json_file)
    products=data['shopify']['productos']
    products.append(product)

    with open(jsonProductsPath,"w",encoding="utf-8") as json_file:
        json.dump(data,json_file,indent=4,ensure_ascii=False)
def get_product_embedding(product:dict)->None:
    r=search_product_embedding(product)
    if r:
        print("Existe, reusando...")
        product['embeding']=r['embeding']
    else:
        print("no existe,agrenado...")
        product['embeding']=get_openai_embedding(str(product))
        add_product(product)
    return product

def get_top_n_match(product:dict,currentCollections:list,n:int)->list:
    with open(jsonCollectionsPath,"r",encoding="utf-8") as json_file:
        data=json.load(json_file)
    saveCollections=data['shopify']['collecciones']
    saveCollectionsNames=[x['Nombre collecion'] for x in saveCollections]
    product=get_product_embedding(product)
    product_embed=product['embeding']
    similarities=[]
    for currentColl in currentCollections:
        if currentColl in saveCollectionsNames:
            pass
        else:
            print("no existe,agrenado...")
            newColl={"Nombre collecion":currentColl,
                     "embeding":get_openai_embedding(currentColl)}
            saveCollections.append(newColl)
            #data['shopify']['collecciones'].append(newColl)
    
    with open(jsonCollectionsPath,"w",encoding="utf-8") as json_file:
        json.dump(data,json_file,indent=4,ensure_ascii=False)

    for collec in saveCollections:
        nameCollection=collec['Nombre collecion']
        embedCollection=collec['embeding']
        similarities.append(
            {"collecion":nameCollection,
             "similarity":get_similarity(product_embed,embedCollection)
             })
    similarities.sort(key=lambda x:x['similarity'],reverse=True)
    print(product['clasificacion'])
    print(similarities[:n])
    return similarities[:n]

def test_top_n_match():
    pathSample=r"C:\DanielBots\Bot-DropShipping\src\marketPlacesOrigen\amazon\skus_Amazon\B098WK7CND\data.json"
    with open(pathSample,"r",encoding="utf-8") as json_file:
        productSample=json.load(json_file)
    currentCollections=["Promociones de laptops","Laptops","Laptops para estudiantes","Laptops para trabajo"]
    get_top_n_match(productSample,currentCollections,3)

if __name__ == '__main__':
    get_openai_embeddings(["sopa","tallarin"])