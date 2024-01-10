from openai import OpenAI
import pandas as pd
import csv
import random
from selenium import webdriver
from extractReviews import extractReviews
import time
import os

chrome_options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=chrome_options)

contenttype_file = '~/main/writeContent/contenttype.csv'
phototype_file = '~/main/writeContent/phototype.csv'
contenttype = pd.read_csv(contenttype_file)
phototype = pd.read_csv(phototype_file)

inputCSV = '~/path/input.csv'
outputCSV = '~/path/output.csv'
df = pd.read_csv(inputCSV)

df2 = pd.read_csv(outputCSV, encoding='utf-8')
df2.to_csv(outputCSV, index=False)

output_directory = os.path.dirname(outputCSV)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

with open(outputCSV, 'a', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['uuid', 'content', 'content_types', 'sub_types']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write header only if the file is empty
    if csv_file.tell() == 0:
        csv_writer.writeheader()

column_a_data = []
for num in range(0,len(df)):
    row = df.iloc[num]
    column_a_data.append({'uuid': row['uuid'],
                           'eng_name': row['eng_name'],
                           'addresslv5': row['addresslv5'],
                           'map_url': row['map_url'],
                           'price_lv' : row['price_lv']
    })

existing_urls = set()
for num in range(0, len(df2)):
    row = df2.iloc[num]
    existing_urls.add((row['uuid'], row['content_types'], row['sub_types']))

column_a_data = [row for row in column_a_data if (row['uuid'], 'main', 'introduction') not in existing_urls]

file_path = '/writeContent/apis.txt'
with open(file_path, 'r') as file:
    file_contents = file.readlines()
    apiList = file_contents
    apiList = [x.replace("\n","") for x in apiList]

def main(name,address,reviews,price):
  prompt = (f'Write an introduction:\n Brand Name: {name} \n Information: {reviews[0:2000]} Address: {address} \n Price Level: {price}')

  action = 'Focus on the introduction about the brand in third-person and STOP WRITING about guests or someone talk about them, avoiding negative side. REMEMBER TO ONLY write in within 130-160 words in third-person.'

  return prompt + action
def photos(name,reviews):
  prompt = (f'Write short description about the PHOTOS and the VIEWS, the Landscape AROUND the:\n Brand Name: {name}\n Information: {reviews}')

  action = 'Focus only on positive side of photos and Views, Landscape around, Dont write about anything else. REMEMBER TO ONLY write in within 130-160 words in third-person.'

  return prompt + action
def menu(name,reviews,price):
  prompt = (f'Introduce about 130 words to introdruce people about the menu and food, drink of the:\n Brand Name: {name} \n  Information: {reviews} Price Level: {price} in third-person')

  action = 'Avoiding Negative side of writeContent and focus only on the menu, Food, drink. Dont write about anything else'

  return prompt + action

for row in column_a_data:
    i = 0
    driver.get(row['map_url'])
    time.sleep(2)

    reviewsContent = " ".join(extractReviews(driver).mostCommon(numReviews=10))

    reviews = [[row['uuid'], x , 'reviews', f'review_{num}'] for num, x in enumerate(" ".join(extractReviews(driver).newest(numReviews=10)))]

    api = random.choice(apiList)
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=api,
    )
    try:
        mainContent = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{main(str(row['eng_name']),str(row['addresslv5']),reviewsContent[0:1500],str(row['price_lv']))}",
                }
            ],
            model="gpt-3.5-turbo",
        )
        mainContentResult = [row['uuid'],mainContent.choices[0].message.content,'main', 'introduction']
    except:
        mainContentResult = [row['uuid'],'','main', 'introduction']

    api = random.choice(apiList)
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=api,
    )

    try:
        photosContent = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{photos(row['eng_name'],reviewsContent[1500:])}",
                }
            ],
            model="gpt-3.5-turbo",
        )
        photosContentResult = [photosContent.choices[0].message.content, 'photos', 'introduction']
    except:
        photosContentResult = [row['uuid'],'','main', 'introduction']

    api = random.choice(apiList)
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=api,
    )

    try:
        menuContent = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{menu(row['eng_name'],reviewsContent[0:2500],row['price_lv'])}",
                }
            ],
            model="gpt-3.5-turbo",
        )

        menuContentResult = [row['uuid'],menuContent.choices[0].message.content, 'menu', 'introduction']
    except:
        menuContentResult = [row['uuid'],'','main', 'introduction']

    contents = [mainContentResult,photosContentResult,menuContentResult]
    for review in reviews:
        contents.append(review)

    # Write all results to the CSV file
    with open(outputCSV, 'a', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['uuid', 'content', 'content_types', 'sub_types']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write each result entry to the CSV file
        for result_entry in contents:
            csv_writer.writerow({
                'uuid': row['uuid'],
                'content': result_entry[0],
                'content_types': result_entry[1],
                'sub_types': result_entry[2]
            })
