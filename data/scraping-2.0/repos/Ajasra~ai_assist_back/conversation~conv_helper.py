import os
import openai
import tiktoken
from dotenv import load_dotenv

from cocroach_utils.database_utils import save_error
from cocroach_utils.db_conv import get_conv_by_id

load_dotenv()


def get_conv_id(conv_id, user_id, doc_id):
    """
    Get the conversation conv_id and return a new one if it does not exist
    :param conv_id:
    :return:
    """
    cur_conv = None
    if conv_id is None or conv_id == -1 or conv_id == 0:
        pass
    else:
        conv = get_conv_by_id(conv_id)
        if conv is None:
            save_error("Conversation not found")
            cur_conv = -1
        else:
            cur_conv = conv

    return cur_conv


def num_tokens_from_message(message, model="gpt-3.5-turbo"):

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_message(message, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_message(message, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = len(encoding.encode(message))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def format_response(response_input):
    """
    Format the response
    :param response_input:
    :return:
    """
    data = []
    # check if there are follow up questions regardless of uppercase or lowercase
    if "FOLLOW UP QUESTIONS:" in response_input:
        data = response_input.split("FOLLOW UP QUESTIONS:")
    elif "FOLLOWUP QUESTIONS:" in response_input:
        data = response_input.split("FOLLOWUP QUESTIONS:")
    elif "Follow up questions:" in response_input:
        data = response_input.split("Follow up questions:")
    elif "Followup questions:" in response_input:
        data = response_input.split("Followup questions:")
    elif "follow up questions:" in response_input:
        data = response_input.split("follow up questions:")
    elif "followup questions:" in response_input:
        data = response_input.split("followup questions:")
    elif "Followup" in response_input:
        data = response_input.split("Followup:")
    elif "FOLLOWUP" in response_input:
        data = response_input.split("FOLLOWUP:")
    elif "followup" in response_input:
        data = response_input.split("followup:")
    elif "follow-up" in response_input:
        data = response_input.split("follow-up:")

    if len(data) > 1:
        # answer = data[0].strip().replace("ANSWER:", "")
        # answer = data[0].strip().replace("answer:", "")
        answer = data[0].strip().replace("Answer:", "")
        follow_up_questions = data[1].strip().split("\n")
        if len(follow_up_questions) == 1:
            follow_up_questions = data[1].strip().split("?")
        return {
            "answer": answer,
            "follow_up_questions": follow_up_questions
        }
    else:
        return {
            "answer": response_input.replace("ANSWER:", "").strip(),
            "follow_up_questions": [],
            "source": ""
        }


def moderation(text):
    print("Moderation text: ", text)
    result = openai.Moderation.create(input=text, api_key=os.getenv("OPENAI_API_KEY"))
    print("Moderation result: ", result)
    result = result["results"][0]
    return result.flagged == True
