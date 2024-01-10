import logging

import numpy as np

import openai


from app.vectorizers.sentence_transformer import model as vectorizer
from app.redis_manager.redis_connector import get_booking_query


def recommend_booking(
        message_history,
        user_input: str,
        interaction_count: int,
        booking_chain,
        context_entities: dict,
        redis_connector,
        city_code
):
    interaction_count += + 1
    logging.warning(interaction_count)
    if interaction_count == 1:
        message_history = []
        # Run the chain only specifying the input variable.
        keywords = booking_chain.run(
            user_input + ', in city: ' + context_entities['CITY'] + ', these are reservation specification: ' + str(context_entities)
        )
        logging.warning(
            user_input + ', in city: ' + context_entities['CITY'] + ', these are reservation specification: ' + str(context_entities)
        )
        logging.warning(keywords)
        logging.warning(city_code)

        top_k = 3
        # vectorize the query
        query_vector = vectorizer.encode(keywords).astype(np.float32).tobytes()
        params_dict = {"vec_param": query_vector}

        # Execute the query
        results = redis_connector.ft().search(
            get_booking_query(top_k, city_code=city_code), query_params=params_dict
        )
        logging.warning(results)
    else:
        results = {}
    if results:
        user_ask = user_input + ', specification of booking requirements: ' + str(context_entities)
    else:
        user_ask = user_input

    message_history, answer, = recommend_found_bookings(
        message_history,
        results, user_ask
    )
    return message_history, answer, interaction_count


def ask_for_booking_details(message_history: list[dict], user_input: str, booking_known_info: dict):
    if booking_known_info:
        known = list(booking_known_info.keys())
    else:
        known = []

    if not message_history:
        system_content = {
                "role": "system",
                "content": "User is asking to book a room and you are "
                           "generic company room or apartment booking ASSISTANT "
                           "asking USER for information to make the booking,"
            }

        needed_entities = [
            ent_name for ent_name in ['FULL_NAME', 'DATES', 'CITY', 'BUDGET', 'GUEST_COUNT'] if ent_name not in known
        ]

        system_content['content'] += (
            f"ONLY ask him nicely for these information on `{', '.join(needed_entities)}`for his booking."
            f"You ask ONLY about: `{', '.join(needed_entities)}`"
        )
        message_history = [
            system_content,
            {"role": "user", "content": user_input}
        ]
    else:
        message_history.append(
            {"role": "user", "content": user_input}
        )

    choices = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history
    )['choices']

    message_history.append(
        {'role': choices[0]['message']['role'], 'content': choices[0]['message']['content'].strip(" \n")}
    )

    return choices[0]['message']['content'].strip(" \n"), message_history


def recommend_found_bookings(message_history, results, user_input):
    if not message_history:
        system_content = {
                "role": "system",
                "content": "You are a REST API SYSTEM connected to AI booking assistant,"
                           "user gives three booking options, SYSTEM ALWAYS REPEATS SHORTENED DESCRIPTION "
                           "OF OPTIONS AND PRESENTS "
                           "each and ask for users preference in ANSWER,"
                           " SYSTEM provides "
                           "JSON OUTPUT containing USER_CONFIRMED_CHOICE key and LISTING_ID "
                           "if USER_CONFIRMED_CHOICE is True, "
                           "USER_CONFIRMED_CHOICE is True when user chooses one of three options,"
                           "LISTING_ID contains ID of chosen option."
                           "Output:\n"
                           '{'
                           '"ANSWER": "Nice description of the three options.",'
                           '"USER_CONFIRMED_CHOICE": Boolean True or False value '
                           'depending on whether user chose one option, '
                           '"LISTING_ID": LISTING_ID which is the id of the apartment '
                           'if USER_CONFIRMED_CHOICE is True'
                           '}\n'
                           "REST API SYSTEM has consistent output.\n"
                           "REST API SYSTEM ALWAYS OUTPUTS JSON containing an ANSWER from AI booking ASSISTANT and "
                           " USER_CONFIRMED_CHOICE and LISTING_ID INFORMATION BASED ON DESCRIBED FORMAT.\n"
                           "SYSTEM DOESNT ASK QUESTIONS ABOUT THE BOOKING, "
                           "SYSTEM DOES NOT ASK FOR ADDITIONAL INFORMATION, "
                           "SYSTEM only asks for choice confirmation at the beginning and "
                           "SYSTEM ALWAYS GIVES OUTPUT as JSON, see example bellow."
                           "SYSTEM IS A REST API, SYSTEMS OUTPUT FORMAT IS ALWAYS JSON"
                           "\n\n"
                           "OUTPUT Format definition:"
                           '{'
                           '"ANSWER": "Nice description of the three options",'
                           '"USER_CONFIRMED_CHOICE": Boolean True or False value '
                           'based on whether positive confirmation was given by user, '
                           '"LISTING_ID": LISTING_ID which is the id of the apartment ' 
                           'if USER_CONFIRMED_CHOICE is True,'
                           '}\n\n'
                           "Examples:\n"
                           "1. Sentence: I don't know, which to choose.\n"
                           "Output:"
                           '{'
                           '"ANSWER": "Oh I think based on what you were looking for, '
                           'the second one would be the best fit.",'
                           '"USER_CONFIRMED_CHOICE": False, '
                           '"LISTING_ID": None '
                           '}'
                           "\n"
                           "\n"
                           "2. Sentence: I love the sound of the third one.\n"
                           "Output:"
                           '{'
                           '"ANSWER": "That is a great choice, I will record to booking for you right away.", '
                           '"USER_CONFIRMED_CHOICE": True, '
                           '"LISTING_ID": 12677097 '
                           '}'
                           "\n"
                           "\n"
                           "3. Sentence: {}\n"
                           "Output: "
            }

        message_history = [
            system_content,
        ]

    full_result_string = ''
    if results:
        for product in results.docs:
            full_result_string += ' '.join(
                [
                    product.price, f", description:", product.description, " Located in city:",
                    product.city,
                    'ID of this booking is:', product.id,
                    "\n\n\n"
                ]
            )
            logging.warning(str(product))
        message_history.append(
            {"role": "user", "content": 'My accomodation booking options are: ' + full_result_string}
        )
    else:
        message_history.append(
            {
                'role': 'user',
                'content': user_input
            }
        )

    choices = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
        temperature=0.3
    )['choices']

    answer = choices[0]['message']['content'].strip(" \n")
    message_history.append(
        {'role': choices[0]['message']['role'], 'content': answer}
    )

    return message_history, answer


def ask_about_general_requirements_response(message_history: list[dict], user_input: str):
    if not message_history:
        message_history = [
            {
                "role": "system",
                "content": "You are the booking assistant answering touristy questions about "
                           "the city user booked his booking at. "
                           "Start conversation by asking User if he wants to ask for "
                           "recommendations of what to do and see in City of his booking."
            },
            {"role": "user", "content": user_input}
        ]
    else:
        message_history.append(
            {
                'role': 'user',
                'content': user_input
            }
        )

    choices = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
        temperature=0.4
    )['choices']

    answer = choices[0]['message']['content'].strip(" \n")
    message_history.append(
        {'role': choices[0]['message']['role'], 'content': answer}
    )

    return message_history, answer


def get_location_recommendations_response(message_history: list[dict], user_input: str):
    if not message_history:
        system_content = {
            "role": "system",
            "content": "You are a REST API SYSTEM connected to AI booking assistant,"
                       "user tells you what city he got booking it and "
                       "AI booking assistant is asking if he has any specific requirements for his booking. "
                       "Output:\n"
                       "{"
                       "'ANSWER': 'What kind of accommodation you expect for your trip?'"
                       "}\n"
                       "You are connected to nice AI booking ASSISTANT. "
                       "REST API SYSTEM has consistent output.\n"
                       "REST API SYSTEM OUTPUTS JSON containing an ANSWER from AI booking ASSISTANT and.\n"
                       "YOU DONT ASK ANY OTHER SPECIFIC QUESTIONS ABOUT THE BOOKING, "
                       "YOU ALWAYS GIVE OUTPUT as JSON, see example bellow."
                       "\n\n"
                       "OUTPUT Format definition:"
                       "{"
                       "'ANSWER': 'Question if USER has a any additional asks for his booking.',"
                       "}\n\n"
                       "Examples:\n"
                       "1. Sentence: I got trip in city City.\n"
                       "Output:"
                       "{"
                       "'ANSWER': "
                       "'Are there any specific requirements for the accomodation you have for your trip in City?',"
                       "}"
                       "\n"
                       "\n"
                       "2. Sentence: I love the sound of the third one.\n"
                       "Output:"
                       "{"
                       "'ANSWER': "
                       "'That is a great choice, do you have any additional preferences for your booking?', "
                       "}"
                       "\n"
                       "\n"
                       "3. Sentence: {}\n"
                       "Output: "
        }

        message_history = [
            system_content,
        ]
    message_history.append(
        {
            'role': 'user',
            'content': user_input
        }
    )

    choices = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
        temperature=0.4
    )['choices']

    answer = choices[0]['message']['content'].strip(" \n")
    message_history.append(
        {'role': choices[0]['message']['role'], 'content': answer}
    )

    return message_history, answer
