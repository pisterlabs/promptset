import openai
from os import environ, path, listdir
from utils.logger import logger  # needed for hidden logs, do not remove
import utils.functions as functions
import json
from utils.internet_access import *
import replicate
import g4f


# Cant import from functions because og the cycle imports
def load_templates(dir: str) -> dict:
    file_dict = {}
    for language_code in listdir(dir):
        if path.isfile(path.join(dir, language_code)):
            with open(path.join(dir, language_code), "r") as f:
                file_dict[language_code] = f.read()
            continue
        for file_name in listdir(path.join(dir, language_code)):
            if file_name.endswith(".txt"):
                file_path = path.join(dir, language_code, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    if language_code not in file_dict:
                        file_dict[language_code] = {file_name: content}
                    else:
                        file_dict[language_code][file_name] = content
    return file_dict


templates = load_templates("templates\\")

openAI_api_key = environ.get("OPENAI_API_KEY")
if not openAI_api_key:
    ("Failed to load OpenAI API key from environment, exiting...")
    exit()
openai.api_key = openAI_api_key


def extract_text(completion: openai.ChatCompletion) -> str:
    """Extracts text from OpenAI API response"""
    try:
        if hasattr(completion, "choices"):
            return completion.choices[0].message.content
        else:
            result = ""
            for i in completion:
                result += i
            return result.replace("LAMA:", "")
    except:
        return str(completion)


def check_function_call(completion: openai.ChatCompletion) -> bool:
    return bool(completion["choices"][0]["message"].get("function_call"))


def extract_function_call_details(completion: openai.ChatCompletion) -> dict:
    return {
        "name": completion["choices"][0]["message"]["function_call"]["name"],
        "args": json.loads(
            completion["choices"][0]["message"]["function_call"]["arguments"]
        ),
    }


def get_messages_in_official_format(messages):
    """Converts messages kept in Johnny to official format"""
    previous_messages = []
    for m in messages:
        if m[0] == "$FUNCTION$":
            previous_messages.append(m[1])
            continue
        previous_messages.append(
            {
                "role": ("assistant" if m[0] == "$BOT$" else "user"),
                "content": m[1],
                "name": functions.remove_utf8_chars(m[0]),
            }
        )
    return previous_messages


def build_prompt_for_lama(messages):
    lama_prompt = ""
    for m in messages:
        if m[0] == "$BOT$":
            lama_prompt += f"LAMA: {m[1]}"
        else:
            lama_prompt += f"{m[0]}: {m[1]}"
        lama_prompt += "\n"
    lama_prompt += "Based on this conversation answer something as telegram bot group smart companion, do not name yourself and return just answer, it will be sent to chat directly"
    return lama_prompt


def generate_image_dalle(prompt, n, size):
    """Returns links to image"""
    response = openai.Image.create(prompt=prompt, n=n, size=size)
    links = []
    for i in response["data"]:
        links.append(i["url"])
    return links


def get_lama_answer(
    prompt,
    system_prompt,
    temperature=0.9,
    top_p=0.5,
    max_tokens=1024,
):
    output = replicate.run(
        "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
        input={
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
        },
    )

    return output


def create_chat_completion(
    johnny,  # for resetting memory in when server error
    messages: list,
    lang: str = "en",
    system_content: str = None,
    answer_length: int = "short",
    use_functions: bool = False,
    reply: bool = False,  # SYS
    model: str = "gpt-3.5-turbo",
    temperature: int = 0.5,
    top_p: float = 1,
    n: int = 1,
    stream: bool = False,
    stop: str = None,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    use_original_api: bool = False,
) -> openai.ChatCompletion:
    """Creates ChatCompletion
    messages(list): list of dicts, where key is users name and value is his message
    reply(bool): True means GPT will consider last message, False means not, None means system input field will be empty
    """

    # --- Building system content ---
    system_content = f"You are telegram bot @JohnnyAIBot, developed by JohnnyCorp team. Your answers should be {answer_length}, use plenty emojis, dont ask how can u help, suggest plenty ideas and critic other ideas"
    if johnny.chat_id < 0:  # group
        system_content += "Suggest plenty ideas, try to find pros and cons of ideas discussed in chat, take part in conversation, ask questions if needed"
    if model == "gigachat":
        system_content += " You are working on GigaChat, new text model developed by Sber russian company"
    if model == "yandexgpt":
        system_content += " You are working on YandexGPT, new text model developed by Yandex russian company"
    if model == "bard":
        system_content += " You are working on Bard, AI chat bot developed by Google"

    previous_messages = [
        {
            "role": "system",
            "content": system_content,
        }
    ]

    # --- Building messages ---

    previous_messages.extend(get_messages_in_official_format(messages))

    # --- Creating ChatCompletion object ---

    chat_completion_arguments = {
        "model": model,
        "messages": previous_messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": stream,
        "stop": stop,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    if use_functions:
        chat_completion_arguments["functions"] = gpt_functions_description
        chat_completion_arguments["function_call"] = "auto"

    try:
        if (model == "lama") or (model == "vicuna"):
            lama_prompt = build_prompt_for_lama(messages)
            completion = get_lama_answer(
                lama_prompt,
                system_prompt=system_content,
                temperature=temperature,
                top_p=top_p,
            )
            logger.info(f"Lama response:{completion}")
        elif ("gpt" in model) and ("yandex" not in model):
            logger.info("Requesting gpt...")
            completion = get_completion(chat_completion_arguments, use_original_api)
        elif model == "gigachat":
            logger.info("Requesting gpt... (GigaChat)")
            chat_completion_arguments["model"] = "gpt-3.5-turbo"
            del chat_completion_arguments["function_call"]
            del chat_completion_arguments["functions"]
            completion = get_completion(chat_completion_arguments, use_original_api)
        elif model == "yandexgpt":
            logger.info("Requesting gpt... (YandexGPT)")
            chat_completion_arguments["model"] = "gpt-3.5-turbo"
            del chat_completion_arguments["function_call"]
            del chat_completion_arguments["functions"]
            completion = get_completion(chat_completion_arguments, use_original_api)
        elif model == "bard":
            logger.info("Requesting gpt... (Bard)")
            chat_completion_arguments["model"] = "gpt-3.5-turbo"
            del chat_completion_arguments["function_call"]
            del chat_completion_arguments["functions"]
            completion = get_completion(chat_completion_arguments, use_original_api)
    except openai.error.APIError as e:
        logger.error(f"OpenAI API returned an API Error: {e}")
        # functions.send_to_developers(
        #     "❗❗Server error occurred❗❗ Using GPT without functions",
        #     johnny.bot,
        #     environ["DEVELOPER_CHAT_IDS"].split(","),
        # )
        del chat_completion_arguments["functions"]
        del chat_completion_arguments["function_call"]
        johnny.messages_history.pop()  # we are not making new prepared_messages! just removing from actual history to consider this in future
        # johnny.messages_history.clear()
        completion = get_completion(chat_completion_arguments, use_original_api)

    except openai.error.APIConnectionError as e:
        logger.error(f"Failed to connect to OpenAI API: {e}")
        raise e
    except openai.error.RateLimitError as e:
        (f"OpenAI API request exceeded rate limit: {e}")
        johnny.messages_to_be_deleted.append(
            johnny.bot.send_message(
                johnny.message.chat.id, templates[johnny.lang_code]["high_demand.txt"]
            )
        )
        return "[WAIT]"
    else:
        logger.info("success 🎉🎉🎉")

    logger.info(f"API completion object: {completion}")

    return completion


def get_completion(args, use_original):
    if use_original:
        return openai.ChatCompletion.create(**args)
    else:
        response = g4f.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=args["messages"],
            provider=g4f.Provider.ChatgptDemo,
            stream=args["stream"],
        )
        return response


available_functions = {
    "google": google,
    "read_from_link": read_from_link,
}


def get_official_function_response(
    function_name: str, function_args: dict, additional_args: dict = {}
) -> list:
    """Takes function name and arguments and returns official response (dict in list)"""

    function_to_call = available_functions[function_name]
    args = {**function_args, **additional_args}
    function_response = function_to_call(**args)
    return {
        "role": "function",
        "name": function_name,
        "content": str(function_response),
    }


def load_functions_for_gpt():
    global gpt_functions_description
    with open("utils\\gpt_functions_description.json") as f:
        gpt_functions_description = json.load(f)
        return gpt_functions_description


load_functions_for_gpt()


def check_context_understanding(answer):
    """Returns bool - True if model answer assumes model understands context else False"""
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f'This is text models answer: {answer}. Is model saying it doesnt understand context or just saying something like "Hi, how can I help you?"? Answer Yes or No',
            }
        ],
        temperature=0,
        max_tokens=1,
    )
    logger.info(f"Check understanding completion: {completion}")
    return extract_text(completion) == "No"


async def generate_suggestion_for_inline(query, verbose=False):
    prepared_messages = [
        {
            "role": "system",
            "content": "Your answer will be sent to telegram dialog, dont try to speak with user, just make what he asking and assume he is using u to speak with another person, be as short as possible",
        },
        {
            "role": "user",
            "content": query,
        },
    ]
    logger.info("Starting asynchronous request to openAI...")
    completion = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=prepared_messages,
        temperature=1,
        max_tokens=220,  # Approx estimation
    )
    logger.info(
        f"Generated suggestion for inline query. Query:{query}; Suggestion:{extract_text(completion)}"
    )
    return extract_text(completion)


def check_theme_context(answer, theme):
    """Returns bool - True is answer is related to theme, False if not"""
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f'This is text mode\'l answer: "{answer}". Is model saying something about {theme}? Answer Yes or No',
            }
        ],
        temperature=0,
        max_tokens=1,
    )
    logger.info(f"Check about theme completion: {completion}")
    return extract_text(completion) == "Yes"


def improve_img_gen_prompt(start_prompt):
    response = g4f.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Make this prompt ({start_prompt}) for AI image generation a bit verbose and detailed, about 5 sentences. Give right prompt itself without any additional words",
            }
        ],
        provider=g4f.Provider.ChatgptDemo,
        stream=False,
    )
    new_prompt = response
    logger.info(
        f"Image prompt improved from {start_prompt} to {extract_text(new_prompt)}"
    )
    return new_prompt


def speech_to_text(path):
    audio_file = open(path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript.text
