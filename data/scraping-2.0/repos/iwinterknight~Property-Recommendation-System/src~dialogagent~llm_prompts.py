import json
import openai

openai.api_key = "****************************************************"



def create_lm_prompt(state, user_utterance, state_history, OOD):
    llm_prompt = None
    if state:
        if not OOD:
            if state == "lq_bedrooms":
                llm_prompt = "Identify the number of bedrooms the user wants from the given utterance delimited by triple backticks.If the response is an exact number, response should be a json string in the following format :\n{\"response\": NUMBER}\nIf the response is a range, response should be a json string in the following format :\n{\"less_than\": NUMBER, \"greater_than\": NUMBER}\nIf the response is empty or undefined or invalid return a json string in the following format :\n{\"response\": -1}\nIf response is undefined or invalid or empty do not provide reasoning.\n```" + user_utterance + "```"
            elif state == "lq_bathrooms":
                llm_prompt = "Identify the number of bathrooms the user wants from the given utterance delimited by triple backticks.\nIf the response is an exact number, response should be a json string in the following format :\n{\"response\": NUMBER}\nIf the response is a range, response should be a json string in the following format :\n{\"less_than\": NUMBER, \"greater_than\": NUMBER}\nIf the response is empty or undefined or invalid return a json string in the following format :\n{\"response\": -1}\nIf response is undefined or invalid or empty do not provide reasoning.\n```" + user_utterance + "```"
            elif state == "lq_price":
                llm_prompt = "Identify the price / price range the user wants from the given response delimited by triple backticks.\nIf the response is an exact age, response should be a json string in the following format :\n{\"response\": NUMBER}\nIf the response is a range, response should be a json string in the following format :\n{\"less_than\": NUMBER, \"greater_than\": NUMBER}\nIf the response is empty or undefined or invalid return a json string in the following format :\n{\"response\": -1}\nIf response is undefined or invalid or empty do not provide reasoning.\n```" + user_utterance + "```"
            elif state == "lq_area":
                llm_prompt = "Identify the house area the user wants from the given response delimited by triple backticks.\nIf the response is an exact area, response should be a json string in the following format :\n{\"response\": NUMBER}\nIf the response is a range, response should be a json string in the following format :\n{\"less_than\": NUMBER, \"greater_than\": NUMBER}\nIf the response is empty or undefined or invalid return a json string in the following format :\n{\"response\": -1}\nIf response is undefined or invalid or empty do not provide reasoning.\n```" + user_utterance + "```"
            elif state == "lq_house_age":
                llm_prompt = "Identify the house age the user wants from the given response delimited by triple backticks.\nIf the response is an exact age, response should be a json string in the following format :\n{\"response\": NUMBER}\nIf the response is a range, response should be a json string in the following format :\n{\"less_than\": NUMBER, \"greater_than\": NUMBER}\nIf the response is empty or undefined or invalid return a json string in the following format :\n{\"response\": -1}\nIf response is undefined or invalid or empty do not provide reasoning.\n```" + user_utterance + "```"
            elif state == "lq_school_rating":
                llm_prompt = "Identify the number / number range the user wants from the given utterance delimited by triple backticks. Response should be between 1 and 5, otherwise it is invalid, in which case return a json string in the following format :\n{\"response\": -1}\nIf the response if an exact number, it should be a json string in the following format :\n{\"response\": NUMBER}\nIf the response is a range, response should be a json string in the following format :\n{\"less_than\": NUMBER, \"greater_than\": NUMBER}\nIf the response is empty or undefined return :\n{\"response\": -1}\nIf response is empty or undefined or invalid do not provide reasoning.\n```" + user_utterance + "```"
            elif state == "lq_house_flooring":
                llm_prompt = "Identify the flooring type the user wants from the given response delimited by triple backticks.\nResponse should be from the following types:\n{\"hardwood\", \"linoleum\", \"vinyl\"}\n Otherwise it is invalid, in which case return a json string in the following format :\n{\"response\": -1}\nIf the response is a valid type, response should be a json string in the following format :\n{\"response\": TYPE}\nIf the response has more than one valid type, response should be a json string in the following format :\n{\"response\": [TYPE, TYPE]}\nIf the response is empty or undefined return a json string in the following format:\n{\"response\": -1}\nIf response is empty or undefined or invalid do not provide reasoning.\n```" + user_utterance + "```"
            elif state == "lq_sea_proximity":
                llm_prompt = "Identify the utterance type from the given response delimited by triple backticks.\nResponse should be one from the following types:\n{\"seaside\", \"less than hour to commute to sea\">, \"inland\"}\n Otherwise it is invalid, in which case return a json string in the following format :\n{\"response\": -1}\nIf the response is a valid type, response should be a json string in the following format :\n{\"response\": TYPE}\nIf the response is empty or undefined return a json string in the following format:\n{\"response\": -1}\nIf response is empty or undefined or invalid do not provide reasoning.\n```" + user_utterance + "```"
            elif state == "lq_house_public_transport":
                llm_prompt = "Identify the utterance type from the given response delimited by triple backticks.\nResponse should be one from the following types:\n{bus, subway}\n Otherwise it is invalid, in which case return a json string in the following format :\n{\"response\": -1}\nIf the response is a valid type, response should be a json string in the following format :\n{\"response\": TYPE}\nIf the response is empty or undefined return a json string in the following format:\n{\"response\": -1}\nIf response is empty or undefined or invalid do not provide reasoning.\n```" + user_utterance + "```"

    return llm_prompt


def get_llm_response(state, user_utterance, state_history, OOD=False):
    llm_prompt = create_lm_prompt(state, user_utterance, state_history, OOD)
    if not llm_prompt:
        print("LLM Error.\n")
        return "ERROR", None
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.2,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            timeout=5
            # stop="stop"
        )
    except Exception as e:
        print("Exception {}".format(e))
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.2,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            timeout=5
            # stop="stop"
        )
    response_json = response["choices"][0]["message"]["content"]
    response_dict = json.loads(response_json)
    if "response" in response_dict:
        if response_dict["response"] == -1:
            print("LLM Error.\n")
            return "ERROR", None
        return "RESPONSE", response_dict["response"]
    elif "less_than" in response_dict or "greater_than" in response_dict:
        return "RANGE", response_dict


def generate_paraphrase(utterance):
    response = None
    paraphrase_prompt = "Generate paraphrase for : " + utterance
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[{"role": "user", "content": paraphrase_prompt}],
            temperature=0.4,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            timeout=5
            # stop="stop"
        )
    except Exception as e:
        print("Exception {}".format(e))
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[{"role": "user", "content": paraphrase_prompt}],
            temperature=0.4,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            timeout=5
            # stop="stop"
        )
    response = response["choices"][0]["message"]["content"]
    return response


def generate_response(utterance):
    response = None
    paraphrase_prompt = "Generate a brief response for : " + utterance
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[{"role": "user", "content": paraphrase_prompt}],
            temperature=0.4,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            # stop="stop"
        )
        response = response["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)

    return response
