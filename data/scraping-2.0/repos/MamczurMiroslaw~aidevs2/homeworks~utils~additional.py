import requests

from openai import OpenAI
from homeworks.utils.aidevsApi import API_KEY_OPENAI

client = OpenAI(api_key=API_KEY_OPENAI)


# function used in C01L04 to return final moderate flag for input text
def moderate_txt(text_to_moderate, print_response=False):
    response = client.moderations.create(
        text_to_moderate,
    )

    final_flag = response["results"][0]["flagged"]
    categories_flag = response["results"][0]["categories"]
    if print_response:
        print(categories_flag)

    return int(final_flag)  # return 0 if False and 1 if True


# function used in C01L05 to send question as POST not JSON
def send_question(token, question, print_response=False):
    url = "https://zadania.aidevs.pl/task/" + token

    # creating dictionary
    data = {
        "question": question
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # sending request
    response = requests.post(url, data=data, headers=headers)

    if response.status_code == 200:
        response_data = response.json()  # Jeśli odpowiedź jest w formacie JSON
        if print_response:
            print(response_data)
    else:
        print(f"Failed to get response:(. Error code: {response.status_code}")

    return response_data


# function used in C02L02 to find 1 most important word
def find_word_to_filter(txt):
    conversation = [
            {"role": "system", "content": """Twoim zadaniem jest zwrócić tylko imioe z podanego niżej tekstu.
Zwracaj tylko 1 imię. 
Jeśli nie możesz znaleźć imienia zwróć Twoim zdaniem najważniejsze 1 słowo, które jest rzeczownikiem.
Przykład:
zdanie:`Abdon ma czarne oczy, średniej długości włosy i pracuje jako prawnik`
Twoja odpowiedź: 'Abdon'
Zdanie użytkownika jest oznaczone w ###
"""},
        {"role": "user", "content": f"###{txt}###"}
        ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        max_tokens=20
    )

    word = response.choices[0].message.content

    return word


def download_json_data_from_url(url):
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse JSON data
            json_data = response.json()

            print("JSON data downloaded successfully.")
            return json_data
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")


# download arch from unknown news
def download_arch():
    # URL of the JSON file
    url = 'https://unknow.news/archiwum.json'

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON content into a DataFrame
        json_arch = response.json()
        return json_arch
    else:
        print(f"Failed to get response:(. Error code: {response.status_code}")
        print(f"Reason: {response.reason}")


# Function to calculate OpenAI Ada-02 embeddings
def calculate_embeddings(text):
    # Call the OpenAI API to get the embedding
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    # return generated embeddings
    return response.data[0].embedding


# function used in C04L01 to download actual currency for specific value
def currency_rate(code="USD"):
    url = f"http://api.nbp.pl/api/exchangerates/rates/A/{code}/"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        currency = data['rates'][0]['mid']
        print(f'I have value for {code}: {currency}')
        return currency
    else:
        print(f"Failed to get response:(. Error code: {response.status_code}")
        print(f"Reason: {response.reason}")
        print(f"Text: {response.text}")


# function used in C04L01 to download country informations
def country_population(country="poland"):
    url = f"https://restcountries.com/v3.1/name/{country}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        pop = data[0]['population']
        print(f'I have value for {country}: {pop}')
        return pop
    else:
        print(f"Failed to get response:(. Error code: {response.status_code}")
        print(f"Reason: {response.reason}")
        print(f"Text: {response.text}")
