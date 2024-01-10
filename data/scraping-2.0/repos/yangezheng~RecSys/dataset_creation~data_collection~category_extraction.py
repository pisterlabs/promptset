import openai
from deep_translator import GoogleTranslator
import json
from tqdm import tqdm
openai.api_key="" #Your API Key

with open(f"data/all_cities_final_last4.json", "r") as fp:
    data = json.load(fp)


with open("system.txt", "r") as fp:
    system_msg = fp.read()

def get_categories(query):
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": query}],
        # functions= function_poi_category_extraction,
        # function_call="auto",
        temperature=1
    )

    response_message = chat_completion["choices"][0]["message"]    
    return eval(response_message["content"])

for poi in tqdm(data):
    city = poi["city"]
    poi_name = poi["name"]
#   if len(poi["category"]) == 0:
    poi_name = GoogleTranslator(source='auto', target='en').translate(poi_name)
    poi_category_set = set()
    for i in range(2): # temperature 1 to extract as much as appropriate category as possible
        # try:
        category_list = get_categories(f"{city}, {poi_name}")["category_list"]
        poi_category_set.update(category_list)
        # except:
        #   poi_category_set.update([])
    poi["categories"] = list(poi_category_set)
  

with open(f"data/all_cities_final_last5.json", "w") as fp:
    fp.write(json.dumps(data, indent=4))
    