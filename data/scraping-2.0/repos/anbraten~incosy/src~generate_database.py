import yaml
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
import requests

load_dotenv()


def sanitize_path(file):
    # replace space with _ and lower case
    return file.replace(' ', '_').lower()


def generate_description(product, qa):
    name = product['name']
    return qa.run("Give me a description of " + name + " in english.")


def generate_use_cases(product, qa):
    name = product['name']
    problem = product['problem']
    return qa.run("What are the use cases of " + name + " in english.")


def generate_funding(product, qa):
    name = product['name']
    description = product['description']
    return qa.run("Generate a text how can I fund " + name + " with following description: " + description + " in english.")


def generate_reviews(product, qa):
    name = product['name']
    return qa.run("Make up a list of reviews from caregivers of a nursing home for " + name + " formatted as markdown list in english.")


def generate_db():
    # load yaml file
    file = open("database.yml")
    database = yaml.load(file, Loader=yaml.FullLoader)
    for product in database['products']:
        print('\n### ' + product['name'] + ' ###')

        filename = sanitize_path(product['name'] + '.txt')
        if not os.path.exists('data/' + filename):
            print(filename+' is missing!')
            continue

        loader = TextLoader("data/" + filename)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(texts, embeddings)

        # qa = VectorstoreIndexCreator().from_loaders([loader])

        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

        if not 'description' in product or product['description'] == '':
            product["description"] = generate_description(product, qa)
            print("Generated description")

        if not 'use_cases' in product or product['use_cases'] == '':
            product["use_cases"] = generate_use_cases(product, qa)
            print("Generated use_cases")

        if not 'funding' in product or product['funding'] == '':
            product["funding"] = generate_funding(product, qa)
            print("Generated funding")

        if not 'reviews' in product or product['reviews'] == '':
            product["reviews"] = generate_reviews(product, qa)
            print("Generated reviews")

        product["description"] = product["description"].strip()
        product["use_cases"] = product["use_cases"].strip()
        product["funding"] = product["funding"].strip()
        product["reviews"] = product["reviews"].strip()

        # write yaml file
        with open('database.yml', 'w') as file:
            yaml.dump(database, file, allow_unicode=True)


def upload_database():
    baseUrl = "https://budibase.app/api/public/v1/"
    url = baseUrl + "tables/" + \
        os.getenv("BUDIBASE_PRODUCTS_TABLE_ID") + "/rows"

    # load yaml file
    file = open("database.yml")
    database = yaml.load(file, Loader=yaml.FullLoader)
    for product in database['products']:
        print('\n### ' + product['name'] + ' ###')

        data = {
            "name": product.get('name'),
            "description": product.get('description'),
            "use_cases": product.get('use_cases'),
            "funding": product.get('funding'),
            "reviews": product.get('reviews'),
            "website": product.get('websites', [""])[0],
        }

        headers = {
            'x-budibase-app-id': os.getenv("BUDIBASE_APP_ID"),
            'x-budibase-api-key': os.getenv("BUDIBASE_API_KEY"),
        }

        response = requests.post(url, json=data, headers=headers)
        print(response.status_code)


if __name__ == "__main__":
    # generate_db()
    upload_database()
