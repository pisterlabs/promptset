# Define a function to simulate the chatbot responses
import os

import dotenv
import openai

from webscraper import call_nhs_search

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def chatbot_response(user_input):
    # In a real chatbot, you would implement the logic to generate responses here.
    # For this example, let's just echo the user's input.
    return f"You said: {user_input}"


def initial_chatbot_response(user_input):
    # completion = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     temperature=0.3,
    #     messages=[
    #         {"role": "system", "content": "You are a consultant at a hospital. You are talking to a patient about their diagnosis. Analyse the patient's diagnosis and suggest treatments."},
    #         {"role": "user", "content": user_input},
    #     ]
    # )

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=10,
        messages=[
            {"role": "system",
             "content": "Return a 4 word search query to find treatments from inputted diagnosis."},
            {"role": "user", "content": user_input},
        ]
    )
    print(completion)
    print(completion["choices"][0]["message"]["content"])
    # search nhs database for treatments of user diagnosis
    extracted_treatment_pages = call_nhs_search(completion["choices"][0]["message"]["content"])
    print(extracted_treatment_pages)
    # take returned list of urls and scrape them for relevant information

    # return relevant information to user
    # TODO: Prepend "Information has been gathered by an automated AI model, please consider consulting a doctor for more information."
    # TODO: Append "Would you like to know more about this treatment?"

    return completion["choices"][0]["message"]["content"]
# lung cancer treatnebt options
