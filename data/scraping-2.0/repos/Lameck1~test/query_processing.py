import openai
import requests
from data_processing import get_openai_api_key

def generate_keywords_with_chatgpt(question):
    openai.api_key = get_openai_api_key()
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Return only the two best keywords from this question: {question}",
        max_tokens=50
    )
    generated_text = response.choices[0].text.strip()
    return generated_text

def extract_intent_and_respond(user_query):
    openai.api_key = get_openai_api_key()
    # Create a prompt to ask ChatGPT for intent
    prompt = f"Identify and return only the main subject/topic in the following query:{user_query}'."

    # Generate a response from ChatGPT
    query_subject_response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )

    main_subject = query_subject_response.choices[0].text.replace(".", "").strip()

    response = f"You might find information on {main_subject} on these files. Select one of them before asking questions on the data:"
        
    return response

def formualte_question(statement):
    openai.api_key = get_openai_api_key()
    # Create a prompt to ask ChatGPT for extracting a question from a statment
    prompt = f"Generate a very short and contectual question from the following statement: {statement}"

    # Generate a response from ChatGPT
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )

    return response.choices[0].text.strip()


def search_hdx_database(keywords):
    url = f'https://data.humdata.org/api/3/action/package_search?q={keywords}'
    print(url)
    response = requests.get(url)
    data = response.json()['result']['results']
    results = []
    for result in data:
        download_url = result['resources'][0]['download_url']
        # Check if the download URL ends with a supported file extension
        if download_url.lower().endswith(('.csv')):
            results.append(
                {'id': result['id'],
                 'title': result['title'],
                 'description': result['resources'][0]['description'],
                 'download_url': download_url}
            )
    return results


