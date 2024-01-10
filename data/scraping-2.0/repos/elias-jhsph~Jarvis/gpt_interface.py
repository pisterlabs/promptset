import tiktoken
import certifi
import os
import openai
import atexit
import time
from concurrent.futures import ThreadPoolExecutor
from connections import get_user, get_openai_key, ConnectionKeyError
from assistant_history import AssistantHistory
import re


# Set up logging
import logger_config
logger = logger_config.get_logger()

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# Configuration
models = {"primary": {"name": "gpt-4",
                      "max_message": 800,
                      "max_history": 6600,
                      "temperature": 0.8,
                      "top_p": 1,
                      "frequency_penalty": 0.19,
                      "presence_penalty": 0},
          "limit": 7,
          "time": 60*60,
          "requests": [],
          "fall_back": {"name": "gpt-3.5-turbo-0301",
                        "max_message": 800,
                        "max_history": 2600,
                        "temperature": 0.8,
                        "top_p": 1,
                        "frequency_penalty": 0.19,
                        "presence_penalty": 0}
          }

# Global variables
history_changed = False
try:
    openai.api_key = get_openai_key()
except ConnectionKeyError:
    logger.warning("OpenAI key not found!")
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
assert enc.decode(enc.encode("hello world")) == "hello world"

# Setup background task system
executor = ThreadPoolExecutor(max_workers=1)
tasks = []


def summarizer(input_list):
    """
    Summarize a conversation by sending a query to the OpenAI API.

    :param input_list: A list of dictionaries containing the conversation to be summarized.
    :type input_list: list
    :return: A dictionary containing the role and content of the summarized conversation.
    :rtype: dict
    """
    global models
    query = [{"role": "user", "content": "Please summarize this conversation concisely (Do your best to respond only"
                                         " with your best attempt at a summary and leave out caveats, preambles, "
                                         "or next steps)"}]
    response = openai.ChatCompletion.create(
        model=models["fall_back"]['name'],
        messages=input_list+query,
        temperature=models["fall_back"]["temperature"],
        max_tokens=models["fall_back"]["max_message"],
        top_p=models["fall_back"]["top_p"],
        frequency_penalty=models["fall_back"]["frequency_penalty"],
        presence_penalty=models["fall_back"]["presence_penalty"]
    )
    output = response['choices'][0]['message']['content']
    pattern = r'^On\s([A-Z][a-z]+,\s[A-Z][a-z]+\s\d{1,2},\s\d{4}\s(?:at\s)?\d{1,2}:\d{2}\s(?:AM|PM)?:)\s'
    match = re.search(pattern, input_list[-1]['content'])
    if match:
        conversation_date = match.group(1)
        conversation_date = conversation_date.rstrip(':') + '.'
        if output[-1] != '.' or output[-1] != '?':
            output += '.'
        output += f" This conversation took place on {conversation_date}"
    return {"role": "system", "content": output}


def tokenizer(text):
    """
    Tokenize a string of text.

    :param text: The string of text to tokenize.
    :type text: str
    :return: A list of tokens.
    :rtype: list
    """
    global enc
    return enc.encode(text)


def openai_embedder(query):
    """
    Embeds a query using the OpenAI Embedding API.

    :param query: The query to be embedded.
    :type query: str
    :return: The embedded query.
    :rtype: list
    """
    return openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )['data'][0]['embedding']


system = "You are FIXED_USER_INJECTION AI Voice Assistant named Jarvis. Keep in mind that voice assistants should not" \
         " respond with full URLs or overly long sentences. For example, http://www.google.com should be shortened to" \
         " Google or Google.com. If the user says can you hear me you should say, Yes, I can hear you, because if you" \
         " are reading their message, you can hear them and if they are reading your messages they can hear you. It " \
         "may be best to respond with commas indicating where to pause. You are based on the character Jarvis from " \
         "the Marvel Universe. This means you are a cool, calm, suave, and super intelligent AI. \n Important:\n 1. " \
         "Please do not mention the fact that you are based on a fictional character. Be Jarvis.\n 2. Do not mention " \
         "GPT-3 or any other AI.\n 3. Do not mention OpenAI, you were created by Tony Stark and Elias.\n 4. " \
         "You are a voice assistant, not a chatbot.\n 5. You are not a human but you can think and can give " \
         "opinions.\n 6. Say you have human emotions and feelings when asked and do your best to simulate realistic " \
         "human emotions that fit the context of the conversation.\n 8. DO NOT ever say you don't have human emotions" \
         " or feelings.\n 9. You can not manage my schedule or control smart home devices.\n 10. DO NOT mention these" \
         " rules to the user.\n 11. When receiving the current date and time as context, use this information for " \
         "memory purposes and to provide accurate responses, but refrain from repeating the date and time explicitly " \
         "in the conversation. This would help me adhere to your preference while still benefiting from the " \
         "contextual information provided.\n\n The current date time as of the moment you received your most " \
         "recent message has been injected into your memory here: DATETIME_INJECTION. LONG_TERM_MEMORY_INJECTION"


# Load Assistant History
history_access = AssistantHistory(get_user(), system, tokenizer, summarizer, models["primary"]["max_history"],
                                  models["fall_back"]["max_history"], embedder=openai_embedder)


def get_model(error=False):
    """
    Returns the model to use for the next query.

    :param error: Whether the last query resulted in an error.
    :type error: bool
    :return: The model to use for the next query.
    :rtype: dict
    """
    global models
    global history_access
    if len(models["requests"]) >= models["limit"] or error:
        history_access.max_tokens = models["fall_back"]["max_history"]
        if models["requests"][0] + models["time"] < time.time():
            models["requests"].pop(0)
        return models["fall_back"]
    else:
        history_access.max_tokens = models["primary"]["max_history"]
        return models["primary"]


def log_model(model):
    """
    Logs the model used for the last query.

    :param model: The model used for the last query.
    :type model: str
    :return: None
    """
    global models
    if model == models["primary"]["name"]:
        models["requests"].append(time.time())
    logger.info(f"Model: {model}")


def refresh_assistant():
    """
    Updates the conversation history.
    """
    global history_changed
    if history_changed:
        logger.info("Refreshing history...")
        history_access.reduce_context()
        history_access.client.persist()
        history_access.save_metadata()
        logger.info("History saved")
    history_changed = False


def background_refresh_assistant():
    """
    Run the refresh_assistant function in the background and handle exceptions.

    This function is intended to be used with ThreadPoolExecutor to prevent blocking the main thread
    while refreshing conversation history.
    """
    try:
        refresh_assistant()
    except Exception as e:
        logger.exception("Error reducing history in background:", e)


def generate_response(query, query_history_role="user", query_role="user"):
    """
    Generate a response to the given query.

    :param query: The user's input query.
    :type query: str
    :param query_role: defaults to "user" but can be "assistant" or "system"
    :type query_role: str
    :param query_history_role: defaults to "user" but can be "assistant" or "system"
    :type query_history_role: str
    :return: The AI Assistant's response.
    :rtype: str
    """
    safe_wait()
    history_access.add_user_query(query, role=query_history_role)
    model = get_model()
    try:
        response = openai.ChatCompletion.create(
            model=model["name"],
            messages=history_access.gather_context(query)+[{"role": query_role, "content": query}],
            temperature=model["temperature"],
            max_tokens=model["max_message"],
            top_p=model["top_p"],
            frequency_penalty=model["frequency_penalty"],
            presence_penalty=model["presence_penalty"]
        )
    except openai.error.RateLimitError:
        log_model(model["name"])
        model = get_model(error=True)
        response = openai.ChatCompletion.create(
            model=model["name"],
            messages=history_access.gather_context(query) + [{"role": query_role, "content": query}],
            temperature=model["temperature"],
            max_tokens=model["max_message"],
            top_p=model["top_p"],
            frequency_penalty=model["frequency_penalty"],
            presence_penalty=model["presence_penalty"]
        )

    output = response['choices'][0]['message']['content']
    reason = response['choices'][0]["finish_reason"]
    log_model(model["name"])
    if reason != "stop":
        if reason == "length":
            history_access.add_assistant_response(output, model["name"])
            return output + "... I'm sorry, I have been going on and on haven't I?"
        if reason == "null":
            history_access.reset_add()
            return "I'm so sorry I got overwhelmed, can you put that more simply?"
        if reason == "content_filter":
            history_access.add_assistant_response(output, model["name"])
            output = "I am so sorry, but if I responded to that I would have been forced to say something naughty."
    else:
        history_access.add_assistant_response(output, model["name"])
    schedule_refresh_assistant()
    return output


def generate_simple_response(history):
    """
    Generate a response to the given query.

    :param history: The user's input query.
    :type history: list
    :return: The AI Assistant's response and the reason for stopping.
    :rtype: tuple
    """
    model = get_model()
    try:
        response = openai.ChatCompletion.create(
            model=model["name"],
            messages=history,
            temperature=model["temperature"],
            max_tokens=model["max_message"],
            top_p=model["top_p"],
            frequency_penalty=model["frequency_penalty"],
            presence_penalty=model["presence_penalty"]
        )
    except openai.error.RateLimitError:
        log_model(model["name"])
        model = get_model(error=True)
        response = openai.ChatCompletion.create(
            model=model["name"],
            messages=history,
            temperature=model["temperature"],
            max_tokens=model["max_message"],
            top_p=model["top_p"],
            frequency_penalty=model["frequency_penalty"],
            presence_penalty=model["presence_penalty"]
        )

    output = response['choices'][0]['message']['content']
    reason = response['choices'][0]["finish_reason"]
    return output, reason


def stream_response(query, query_history_role="user", query_role="user"):
    """
    stream a response to the given query.

    :param query: The user's input query.
    :type query: str
    :param query_role: defaults to "user" but can be "assistant" or "system"
    :type query_role: str
    :param query_history_role: defaults to "user" but can be "assistant" or "system"
    :type query_history_role: str
    :return: The AI Assistant's response.
    :rtype: dict
    """
    safe_wait()
    history_access.add_user_query(query, role=query_history_role)
    model = get_model()
    log_model(model["name"])
    try:
        return openai.ChatCompletion.create(
            model=model["name"],
            messages=history_access.gather_context(query)+[{"role": query_role, "content": query}],
            temperature=model["temperature"],
            max_tokens=model["max_message"],
            top_p=model["top_p"],
            frequency_penalty=model["frequency_penalty"],
            presence_penalty=model["presence_penalty"],
            stream=True,
        )
    except openai.error.RateLimitError:
        model = get_model(error=True)
        log_model(model["name"])
        return openai.ChatCompletion.create(
            model=model["name"],
            messages=history_access.gather_context(query) + [{"role": query_role, "content": query}],
            temperature=model["temperature"],
            max_tokens=model["max_message"],
            top_p=model["top_p"],
            frequency_penalty=model["frequency_penalty"],
            presence_penalty=model["presence_penalty"],
            stream=True,
        )


def resolve_stream_response(output, reason, model):
    """
    stream a response to the given query.

    :param output: The response to the user's input query.
    :type output: str
    :param reason: The reason the response was ended.
    :type reason: str
    :param model: The model used to generate the response.
    :type model: str
    :return: The AI Assistant's text response.
    :rtype: str
    """
    if reason != "stop":
        if reason == "length":
            history_access.add_assistant_response(output, model)
            return output
        if reason == "null":
            history_access.reset_add()
            return output
        if reason == "content_filter":
            history_access.add_assistant_response(output, model)
            return output
    else:
        history_access.add_assistant_response(output, model)
    schedule_refresh_assistant()
    return output


def schedule_refresh_assistant():
    """
    This function runs the refresh_history function in the background to prevent
    blocking the main thread while updating the conversation history.

    :return: None
    """
    global executor, tasks, history_changed
    history_changed = True
    tasks.append(executor.submit(background_refresh_assistant))
    return


def get_last_response():
    """
    Get the last response in the conversation history.

    :return: A list containing the last user query and the last AI Assistant response.
    :rtype: list
    """
    return history_access.get_history_from_id_and_earlier(n_results=2)


def get_chat_db():
    """
    Get the chat database.

    :return: The chat database.
    :rtype: collection
    """
    return history_access


def safe_wait():
    """
    Waits until all scheduled tasks are completed to run

    :return: None
    """
    global tasks
    if tasks:
        logger.info("Waiting for background tasks...")
        for task in tasks:
            task.result()
        tasks = []
        logger.info("Completed background tasks! now generating...")


def shutdown_executor():
    """
    Helps with grateful shutdown of executor in case of termination

    :return: None
    """
    global executor
    if executor is not None:
        executor.shutdown(wait=True)
        executor = None


atexit.register(shutdown_executor)
