from translate_class import SpeechToTextTranslator
from openai_class import InformationExtractor
from owlready2 import get_ontology, default_world
from conversation_log import ConversationLogger
from pydub import AudioSegment
from questions import SparqlQueryQuestions
from datetime import datetime
import socket
import new_questions


class SparqlQuery:
    def __init__(self, ontology_path):
        self.ontology = get_ontology(ontology_path).load()

    def run_query(self, query):
        results = default_world.sparql(query)
        for row in results:
            print(row[0].name)

translation_keywords = {
    "en-US": ["translate", "translation", "interpret", "interpreter", "convert"],
    "it-IT": ["tradurre", "traduciamo", "tradurlo", "traduzione",  "interpretare", "interprete", "convertire"],
    "Spanish": ["traducir", "traducción", "interpretar", "intérprete", "convertir"],
    "de-DE": ["übersetzen", "Übersetzung", "dolmetschen", "Dolmetscher", "umwandeln"]
}

def translation_request(transcribed_text, text_to_be_translated, language_code):
    target_language = language_code
    language_code = {'en-US': 'English', 'it-IT': 'Italian', 'de-DE': 'German'}.get(language_code, language_code)
    print(f"language_code: {language_code}")
    # prompt = f"Robot asked Child: '{text_to_be_translated}', Child replied: '{transcribed_text}'. If translation is requested on the child's reply, give the translation of the question/reply('it'/'question' might refer to what robot asked) without asking follow-up questions. If No translation is requested, return just NO (Nothing else and no explanation). "
    prompt = f"Child asked robot: Can you translate {text_to_be_translated}. Translate ONLY(NO OTHER EXPLANATION) '{text_to_be_translated}' in {language_code} "
    # prompt = f"Check if translation is explicitly requested in the following message: {transcribed_text}. If yes, give the translation without asking follow-up questions. If No, return just NO (Nothing else and no explanation). Does this reply ask to translate the previous question? If yes, return YES. If No, return No"
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    # if response.lower() == "no":
    #     return None
    # translate_and_synthesize(og_language, ontology_text=response)
    translator.synthesize_speech(target_language, response)
    send_nao(Nao_text=response, language_code=language_code)
    # return True

def is_translation_request(transcribed_text, og_language,text_to_be_translated):
    for keyword in translation_keywords[og_language]:
        if keyword in transcribed_text:
            print("It's a translation request")
            translation_request(transcribed_text, text_to_be_translated, og_language)
            return True
    return False

def send_nao(Nao_text, language_code):
    language_code = {'en-US': 'English', 'it-IT': 'Italian', 'de-DE': 'German'}.get(language_code, language_code)
    message = f"{Nao_text}|{language_code}"
    client_socket.sendall(message.encode())
    data = client_socket.recv(1024)
    print(data)
    return(data)

def sentence_case(input_string):
    return input_string[0].upper() + input_string[1:].lower()

def translate_and_synthesize(og_language, ontology_text):
    text = translator.translate_text(og_language, ontology_text)
    translator.synthesize_speech(og_language, text)
    send_nao(text, og_language)

def get_response_make_dish(transcribed_text):
    prompt = f"The child was asked: 'Have you ever helped make _____(a dish) at home?'. The child replied: '{transcribed_text}'. Give a reply to the child's answer without asking a question."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    translate_and_synthesize(og_language, ontology_text=response)

def get_response_communication(transcribed_text):
    prompt = f"The child was asked: 'Imagine that in my home, we communicate through light patterns. How do you express yourselves here?'. The child replied: '{transcribed_text}'. Give a reply to the child's answer without asking a question."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    translate_and_synthesize(og_language, ontology_text=response)

def get_response_food(transcribed_text):
    prompt = f"The child was asked: 'What kind of food do you eat in your country?'. The child replied: '{transcribed_text}'. Give a reply to the child's answer without asking a question."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    translate_and_synthesize(og_language, ontology_text=response)

def get_response_try_dish(transcribed_text):
    prompt = f"The child was asked: 'Have you ever tried _____(a dish)?'. The child replied: '{transcribed_text}'. Give a reply to the child's answer WITHOUT asking a question at the end."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    translate_and_synthesize(og_language, ontology_text=response)

def get_response_capital(transcribed_text):
    prompt = f"The child was asked: 'Did you know that the capital of _____(country) is ______(capital)?'. The child replied: '{transcribed_text}'. Give a reply to the child's answer WITHOUT asking a question at the end."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    translate_and_synthesize(og_language, ontology_text=response)

def get_response_animals(transcribed_text):
    prompt = f"The child was asked: 'What kinds of animals do you have here, and what do you do with them?'. The child replied: '{transcribed_text}'. Give a reply to the child's answer WITHOUT asking a question at the end."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    translate_and_synthesize(og_language, ontology_text=response)

def get_response_sport(transcribed_text):
    prompt = f"The child was asked: 'What sports do you like?'. The child replied: '{transcribed_text}'. Return ONLY the name of the sport from child's reply."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    return sentence_case(response)

def get_response_sport2(transcribed_text):
    prompt = f"The child was asked: 'Do you know that the main players of _____ (sport) in _____(country) are ____(player names)?'. The child replied: '{transcribed_text}'. Give a reply to the child's answer WITHOUT asking a question at the end."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    translate_and_synthesize(og_language, ontology_text=response)

def get_response_adventure(transcribed_text):
    prompt = f"The child was asked: 'What would you wish for, if you could have any adventure in the universe?'. The child replied: '{transcribed_text}'. Give a reply to the child's answer WITHOUT asking a question at the end."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    translate_and_synthesize(og_language, ontology_text=response)


def get_response_festivals(transcribed_text):
    prompt = f"The child was asked: 'Have you ever participated in any of these festivals?'. The child replied: '{transcribed_text}'. Give a reply to the child's answer WITHOUT asking a question at the end."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    translate_and_synthesize(og_language, ontology_text=response)

def get_response_fun(transcribed_text):
    prompt = f"The child was asked: 'What do you do for fun in your country?'. The child replied: '{transcribed_text}'. If the child mentions something about sports, Return a 'Yes'. Else, give a suitable reply to the child's answer WITHOUT asking a question at the end."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    if response.lower() != "yes." and response.lower()!= "yes":
        print(response)
        translate_and_synthesize(og_language, ontology_text=response)
    else:
        return response.lower()

def get_response_yes_or_no(transcribed_text):
    prompt = f"The child was asked: 'Would you like to learn a few simple phrases in ____ (language). The child replied: '{transcribed_text}'. Return a 'Yes' or 'No' based on the child's answer"
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    translate_and_synthesize(og_language, ontology_text=response)
    return response.lower()

def get_country(transcribed_text):
    prompt = f"Which country would someone be if he says: {transcribed_text}. Output just the country name"
    return information_extractor.extract_information(transcribed_text, prompt, temperature=0)

def user_translation(transcribed_text):
    prompt = f"Robot asked child: 'What would you like to translate and to which language do you want to translate it?'. The child replied: {transcribed_text}. Give the response without asking follow-up questions."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0)
    print(response)
    translate_and_synthesize(og_language, ontology_text=response)

def collaborative_story(transcribed_text, next_question):
    prompt = f"Given the previous response ‘{transcribed_text}’, please modify the following question ‘{next_question}’ such that it is related to the previous response and also maintain the flow of a collaborative storytelling. Remember to use the person’s name only at the end of the question and keep the original intent of the question intact."
    response = information_extractor.extract_information(transcribed_text, prompt, temperature=0.5)
    print(response)
    translate_and_synthesize(og_language, ontology_text=response)    
    

project_id = "resonant-tract-404715"
language_codes = ["it-IT"]
audio_file = "output.wav"

information_extractor = InformationExtractor()
translator = SpeechToTextTranslator(project_id, language_codes, audio_file)
sparql_query = SparqlQueryQuestions("/home/jerin/robotics/Thesis/pedagogy_ontology_v2.rdf")
logger = ConversationLogger('conversation.log')

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 12345))

time = "Morning" if 6 <= datetime.now().hour < 18 else "Evening"

# Hello there, little Earthlings!
text = new_questions.get_question(0)
print(text)
language_code = "en-US"
translator.synthesize_speech(language_code, text)

receipt1 = send_nao(text, language_code)

while True:
    translator.record(audio_file)
    transcribed_text, og_language = translator.transcribe_multiple_languages_v2()

    # transcribed_text = input("Type your response: ")
    # og_language = input("Write your language code: ")
    logger.log_message('Child', transcribed_text)

    translation_request_result = is_translation_request(transcribed_text, og_language,text_to_be_translated=text)

    if translation_request_result is False:
        country = get_country(transcribed_text)

        country_capital_text = sparql_query.get_country_capital(country, time)
        print(country_capital_text)
        translate_and_synthesize(og_language, country_capital_text)

        while True:
            translator.record(audio_file)
            transcribed_text, og_language = translator.transcribe_multiple_languages_v2()

            # transcribed_text = input("Type your response: ")
            # og_language = input("Write your language code: ")
            logger.log_message('Child', transcribed_text)

            translation_request_result = is_translation_request(transcribed_text, og_language,text_to_be_translated=country_capital_text)

            if translation_request_result is False:

                get_response_capital(transcribed_text)

                break

                

        # ontology_text, random_food = sparql_query.run_query(country, time)
        # print(ontology_text)
        # translate_and_synthesize(og_language, ontology_text)
        break



# How do you express yourselves?
text = new_questions.get_question(1)
collaborative_story(transcribed_text, next_question=text)

while True:
    translator.record(audio_file)
    transcribed_text, og_language = translator.transcribe_multiple_languages_v2()
    # transcribed_text = input("Type your response: ")
    # og_language = input("Write your language code: ")
    logger.log_message('Child', transcribed_text)
    translation_request_result = is_translation_request(transcribed_text, og_language,text_to_be_translated=text)


    if translation_request_result is False:
        get_response_communication(transcribed_text)
        break


# What kind of food do you eat in your country?
text = new_questions.get_question(2)
collaborative_story(transcribed_text, next_question=text)

while True:
    translator.record(audio_file)
    transcribed_text, og_language = translator.transcribe_multiple_languages_v2()
    # transcribed_text = input("Type your response: ")
    # og_language = input("Write your language code: ")
    logger.log_message('Child', transcribed_text)
    translation_request_result = is_translation_request(transcribed_text, og_language,text_to_be_translated=text)


    if translation_request_result is False:
        get_response_food(transcribed_text)

        ontology_text, random_food = sparql_query.run_query(country, time)
        print(ontology_text)
        translate_and_synthesize(og_language, ontology_text)
        break


# ingredients = sparql_query.get_ingredients(random_food)
# text = sparql_query.generate_question(random_food, ingredients, country)
# print(text)
# translate_and_synthesize(og_language, ontology_text=text)

while True:
    translator.record(audio_file)
    transcribed_text, og_language = translator.transcribe_multiple_languages_v2()
    # transcribed_text = input("Type your response: ")
    # og_language = input("Write your language code: ")
    logger.log_message('Child', transcribed_text)
    translation_request_result = is_translation_request(transcribed_text, og_language,text_to_be_translated=text)


    if translation_request_result is False:
        get_response_make_dish(transcribed_text)
        break



# What kinds of animals do you have here, and what do you do with them?
text = new_questions.get_question(3)
collaborative_story(transcribed_text, next_question=text)

while True:
    translator.record(audio_file)
    transcribed_text, og_language = translator.transcribe_multiple_languages_v2()
    # transcribed_text = input("Type your response: ")
    # og_language = input("Write your language code: ")
    logger.log_message('Child', transcribed_text)
    translation_request_result = is_translation_request(transcribed_text, og_language,text_to_be_translated=text)


    if translation_request_result is False:
        get_response_animals(transcribed_text)
        break


# og_language = "en-US"
# country = "Italy"
# transcribed_text = "We have dogs, cats, and cows"
# og_language = "en-US"
# country = "Italy"
# What do you do for fun in your country? 
text = new_questions.get_question(4)
collaborative_story(transcribed_text, next_question=text)

while True:
    translator.record(audio_file)
    transcribed_text, og_language = translator.transcribe_multiple_languages_v2()
    # transcribed_text = input("Type your response: ")
    # og_language = input("Write your language code: ")
    logger.log_message('Child', transcribed_text)
    translation_request_result = is_translation_request(transcribed_text, og_language,text_to_be_translated=text)


    if translation_request_result is False:
        get_response = get_response_fun(transcribed_text)
        print(f"Get response: {get_response}")
        if get_response == 'yes' or get_response == 'yes.':
            if country == 'Italy':
                text = "What sports do you like? Do you like Football or Volleyball?"
                print(text)
                translate_and_synthesize(og_language, ontology_text=text)
                translator.record(audio_file)
                transcribed_text, og_language = translator.transcribe_multiple_languages_v2()
                # transcribed_text = input("Type your response: ")
                # og_language = input("Write your language code: ")
                logger.log_message('Child', transcribed_text)
                sport = get_response_sport(transcribed_text)
                text = sparql_query.get_main_players(sport, country)
                print(text)
                translate_and_synthesize(og_language, ontology_text=text)
                translator.record(audio_file)
                transcribed_text, og_language = translator.transcribe_multiple_languages_v2()
                # transcribed_text = input("Type your response: ")
                # og_language = input("Write your language code: ")
                logger.log_message('Child', transcribed_text)
                get_response_sport2(transcribed_text)
            
            elif country == 'Germany':
                text = "What sports do you like? Do you like Football?"
                print(text)
                translate_and_synthesize(og_language, ontology_text=text)
                translator.record(audio_file)
                transcribed_text, og_language = translator.transcribe_multiple_languages_v2()
                # transcribed_text = input("Type your response: ")
                # og_language = input("Write your language code: ")
                logger.log_message('Child', transcribed_text)
                sport = get_response_sport(transcribed_text)
                text = sparql_query.get_main_players(sport, country)
                print(text)
                translate_and_synthesize(og_language, ontology_text=text)
                translator.record(audio_file)
                transcribed_text, og_language = translator.transcribe_multiple_languages_v2()
                # transcribed_text = input("Type your response: ")
                # og_language = input("Write your language code: ")
                logger.log_message('Child', transcribed_text)
                get_response_sport2(transcribed_text)
        else:
            break


        break


# What special celebrations do you have here?
text = new_questions.get_question(5)
collaborative_story(transcribed_text, next_question=text)

while True:
    translator.record(audio_file)
    transcribed_text, og_language = translator.transcribe_multiple_languages_v2()
    # transcribed_text = input("Type your response: ")
    # og_language = input("Write your language code: ")
    logger.log_message('Child', transcribed_text)
    translation_request_result = is_translation_request(transcribed_text, og_language,text_to_be_translated=text)


    if translation_request_result is False:
        text = sparql_query.get_main_festivals(country)
        print(text)
        translate_and_synthesize(og_language, ontology_text=text)
        translator.record(audio_file)
        transcribed_text, og_language = translator.transcribe_multiple_languages_v2()
        # transcribed_text = input("Type your response: ")
        # og_language = input("Write your language code: ")
        logger.log_message('Child', transcribed_text)
        get_response_festivals(transcribed_text)

        break

# What would you wish for, if you could have any adventure in the universe?
text = new_questions.get_question(6)
collaborative_story(transcribed_text, next_question=text)

while True:
    translator.record(audio_file)
    transcribed_text, og_language = translator.transcribe_multiple_languages_v2()
    # transcribed_text = input("Type your response: ")
    # og_language = input("Write your language code: ")
    logger.log_message('Child', transcribed_text)
    translation_request_result = is_translation_request(transcribed_text, og_language,text_to_be_translated=text)


    if translation_request_result is False:
        get_response_adventure(transcribed_text)
        break


text = new_questions.get_question(7)
print(text)
translate_and_synthesize(og_language, ontology_text=text)
