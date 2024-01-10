
import openai
import os
from dotenv import load_dotenv

#------------------------------------#
#            OPENAI CREDS            #
#------------------------------------#

load_dotenv()
openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API")


#------------------------------------#
#              FUNCTIONS             #
#------------------------------------#

def analyze_text_via_gpt(audio_details):

    if audio_details["type"] == "Local":

        if audio_details["length"] > 180000:
            text_to_send = f"""
            W kilku zdaniach opowiedz mi co myslisz o pewnym filmie,
            odpowiedź zacznij słowami 'Ten film opowiada/jest o'
            Na początku filmu autor powiedział: {audio_details["first_words"]}.
            W trakcie filmu padają zdania: {audio_details["middle_words"]}.
            Na koncu filmu autor powiedział: {audio_details["last_words"]}.
            Jeśli nie możesz nic powiedzieć o filmie, napisz, że film ma za mało informacji żeby go opisać
            """

        else:
            text_to_send = f"""
            W kilku zdaniach opowiedz mi co myslisz o pewnym filmie,
            odpowiedź zacznij słowami 'Ten film opowiada/jest o'
            W trakcie filmu padają zdania: {audio_details["text"]}.
            Jeśli nie możesz nic powiedzieć o filmie, napisz, że film ma za mało informacji żeby go opisać
            oraz powiedz, że może film moze ma samą muzykę jeśli nie padają żadne zdania, albo są one niezrozumiałe
            """

    else:

        if audio_details["length"] > 180000:

            text_to_send = f"""
            W kilku zdaniach opowiedz mi co myslisz o pewnym filmie,
            odpowiedź zacznij słowami 'Ten film opowiada/jest o'
            Wiemy ze film ma tagi: {audio_details["tags"]}.
            Wiemy ze opis filmu to: {audio_details["title"]}.
            Na początku filmu autor powiedział: {audio_details["first_words"]}.
            W trakcie filmu padają zdania: {audio_details["middle_words"]}.
            Na koncu filmu autor powiedział: {audio_details["last_words"]}.
            Jeśli nie możesz nic powiedzieć o filmie, napisz, że film ma za mało informacji żeby go opisać
            Jeśli film nie ma tagów to nic nie pisz ani nie wspominaj o tym.
            """

        else:
            text_to_send = f"""
            W kilku zdaniach opowiedz mi co myslisz o pewnym filmie,
            odpowiedź zacznij słowami 'Ten film opowiada/jest o'
            Wiemy ze film ma tagi: {audio_details["tags"]}.
            Wiemy ze opis filmu to: {audio_details["title"]}.
            W trakcie filmu padają zdania: {audio_details["text"]}.
            Jeśli nie możesz nic powiedzieć o filmie, napisz, że film ma za mało informacji żeby go opisać
            oraz powiedz, że może film moze ma samą muzykę jeśli nie padają żadne zdania, albo są one niezrozumiałe
            Jeśli film nie ma tagów to nic nie pisz ani nie wspominaj o tym.
            """

    

    openai.Model.list()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{text_to_send}"}
        ]
    )

    
    audio_details['gpt_response'] = response['choices'][0]['message']['content']

    return audio_details


