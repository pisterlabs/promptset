import json
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

import sys
sys.path.append("api")
from local_creds import *
from query import *

#langchain openai interface
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def get_possible_recommended_products(product_id, count):
    product_vetor = get_product_vector(product_id)
    similar_products = get_similar_products(product_vetor, count)
    return similar_products

def build_full_prompt(product_id, count):
    long_product_list = get_possible_recommended_products(product_id, 8)
    strip_blank_fields = lambda a_dict : {key: a_dict[key] for key in a_dict if a_dict[key]!=""}
    strip_for_query = lambda a_dict : {key: a_dict[key] for key in ("product_name", "brand_name", "category", "selling_price", "about_product", "selling_price", "product_specification", "technical_details", "shipping_weight") }
    stripped_product_list = [strip_blank_fields(strip_for_query(row)) for row in long_product_list]
    string_product_list = ["PRODUCT NUMBER "+str(ind) + ": " + json.dumps(product) for ind, product in enumerate(stripped_product_list)]

    #prompt that is sent to openai using the response from the vector database
    prompt_boilerplate = "Of the following products, all preceded with PRODUCT NUMBER, select the " + str(count) + " products most recommended to shoppers who bought the product preceded by ORIGINAL PRODUCT below. Return the product_id corresponding to those products."
    original_product_section = "ORIGINAL PRODUCT: " + json.dumps(strip_blank_fields(strip_for_query(get_product(product_id))))
    comparable_products_section = "\n".join(string_product_list)
    final_answer_boilerplate = "Final Answer: "
    nl = "\n"
    return (prompt_boilerplate + nl + original_product_section + nl + comparable_products_section + nl + final_answer_boilerplate, long_product_list)

def get_recommended_products(product_id, count):
    full_prompt, products = build_full_prompt(product_id, count)
    result = llm.predict(full_prompt)
    index_list = [int(i) for i in result.split(",")]
    prod_list = [products[i] for i in index_list]
    return prod_list

def embed(text_to_embed):
    embedding = list(embeddings.embed_query(text_to_embed))
    return [float(component) for component in embedding]

def get_search_results(query, count):
    query_vector = embed(query)
    relevant_products = get_similar_products(query_vector, count)
    return relevant_products
