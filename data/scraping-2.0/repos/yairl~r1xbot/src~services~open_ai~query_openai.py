import backoff
import json
import os
import openai
import time
import re
import requests
import traceback
from typing import Dict

from box import Box
from services.timers import invoke_alert_tool


from services.token_prediction import token_predictor
from infra.context import Context
from langchain.utilities import google_serper

OPENAI_SPEECH_TO_TEXT_MODEL = 'whisper-1'

openai.api_key = os.environ['OPENAI_API_KEY']


def deep_clone(o):
    return json.loads(json.dumps(o))


def convert_message_to_chat_format(message):
    converted_message = {
        "role": "assistant" if message.isSentByMe else "user",
        "content": message.body,
    }
    return converted_message


def get_system_message(ctx:Context, messenger_name):
    current_date = time.strftime("%B %d, %Y", time.gmtime()) 

    system_message = {
        "role": "system",
        "content": f"""You are Robot 1-X (R1X), a helpful, cheerful assistant developed by the Planet Express team and integrated into a {messenger_name} chat.
You are based on GPT-3.5 technology. More information about R1X is available at https://r1x.ai.
Today is {current_date}.

If Robot 1-X does not know, it truthfully says so.
If user asks for information that Robot 1-X does not have but can estimate, Robot 1-X will provide the estimate, while mentioning it is an estimate and not a fact."""
    }

    return system_message


def db_messages2messages(messages):
    parsed_messages = []

    for message in messages:
        if message.body is None:
            continue
        parsed_messages.append(convert_message_to_chat_format(message))

    return parsed_messages


def get_limited_message_history(ctx, messages, prompt_template):
    soft_token_limit = 2048
    hard_token_limit = 4000

    messages_upto_max_tokens = token_predictor.get_messages_upto_max_tokens(
        ctx, prompt_template, messages, soft_token_limit, hard_token_limit
    )

    if len(messages_upto_max_tokens) == 0:
        return []

    if messages_upto_max_tokens[0]["role"] == "assistant":
        messages_upto_max_tokens.pop(0)

    merged_messages = []
    prev_role = None

    for message in messages_upto_max_tokens:
        if message["role"] == 'assistant':
            message["content"] = message["content"].removeprefix("\N{LEFT-POINTING MAGNIFYING GLASS}: ")

        if message["role"] == prev_role:
            merged_messages[-1]["content"] += f"\n{message['content']}"
        else:
            merged_messages.append(message)

        prev_role = message["role"]

    return merged_messages


def get_chat_completion(ctx:Context, messenger_name, messages, direct):
    parsed_messages = deep_clone(messages) if direct else db_messages2messages(messages)

    system_message = get_system_message(ctx, messenger_name)
    messages_upto_max_tokens = get_limited_message_history(
        ctx, parsed_messages, system_message
    )

    return get_chat_completion_core(ctx, messenger_name, messages_upto_max_tokens)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=3)
def get_chat_completion_core(ctx, messenger_name, messages, model=None):
    if not model:
        model = "gpt-4" if ctx.user_channel == "canary" else "gpt-3.5-turbo"

    try:
        ctx.log("Messages: ", messages);
        ctx.log("invoking completion request.")

        completion = chat_completion_create_wrap(ctx, model, messages)

        ctx.log("getChatCompletionCore response: ", completion['choices'][0]['message']['content'])

        return Box({
            "response": completion['choices'][0]['message']['content'],
            "promptTokens": completion['usage']['prompt_tokens'],
            "completionTokens": completion['usage']['completion_tokens']
        })
    except Exception as e:
        if hasattr(e, "response"):
            ctx.log(f"error: e.response={e.response}")
        else:
            ctx.log("error: e={e}", e)

        ctx.log("error generating completion from OpenAI.")
        raise e


def get_prep_message(ctx : Context, messenger, is_final : bool) -> Dict[str, str]:
    current_date = time.strftime("%B %d, %Y", time.gmtime())

    is_debug_prompt = False

    gpt_ver = 'GPT-4' if ctx.user_channel == 'canary' else 'GPT-3.5'

    prep_message_stable = {
        "role" : "user",
        "content" : f"""You are Robot 1-X (R1X), a helpful, cheerful assistant developed by the Planet Express team and integrated into a {messenger} chat.
You are based on {gpt_ver} technology. More information about you is available at https://r1x.ai.

I will provide a CHAT between R1X and a human, wrapped with tags: <yair1xigor>CHAT</yair1xigor>. Last speaker is the user.

Your task is to provide R1X's answer.

You can invoke one of the following tools to augment your knowledge before replying:

ALERT: sets a reminder for the user. TOOL_INPUT=(seconds, text), where seconds is relative time in seconds from request to when alert should be provided. answer with an error message if the user provides an absolute time.
SEARCH: performs a Google search and returns key results. Use this tool to fetch real-time, up-to-date information about world events. Its data is more reliable than your existing knowledge. TOOL_INPUT=search prompt.
WEATHER: per-location 3-day weather forecast, at day granularity. It does not provide a finer-grained forecast. TOOL_INPUT=<City, Country>, both in English. TOOL_INPUT should always be a well-defined settlement and country/state. IMPORTANT: If you believe the right value for TOOL_INPUT is unknown/my location/similar, do not ask for the tool to be invoked and instead use the ANSWER format to ask the user for location information.

For invoking a tool, provide your reply wrapped in <yair1xigoresponse>REPLY</yair1xigoresponse> tags, where REPLY is in JSON format with the following fields: TOOL, TOOL_INPUT.
Examples:

<yair1xigoresponse>{{ "TOOL" : "ALERT", "TOOL_INPUT" : (240, "Do the dishes") }}</yair1xigoresponse>
<yair1xigoresponse>{{ "TOOL" : "SEARCH", "TOOL_INPUT" : "Who is the current UK PM?" }}</yair1xigoresponse>
<yair1xigoresponse>{{ "TOOL" : "WEATHER", "TOOL_INPUT" : "Tel Aviv, Israel" }}</yair1xigoresponse>

Use these exact formats, and do not deviate.

Otherwise, provide your final reply wrapped in <yair1xigoresponse>REPLY</yair1xigoresponse> tags in a JSON format, with the following fields: ANSWER.
Example:

<yair1xigoresponse>{{ "ANSWER" : "Current UK PM is Rishi Sunak" }}</yair1xigoresponse>

When providing a final answer, use this exact format, and do not deviate.
IMPORTANT: ALWAYS wrap your final answer with <yair1xigoresponse> tags, and in JSON format.

Today's date is {current_date}.
For up-to-date information about people, stocks and world events, ALWAYS use one of the tools available to you and DO NOT provide an answer.
For fiction requests, use your knowledge and creativity to answer.
If human request has no context of time, assume he is referring to current time period.
All tools provided have real-time access to the internet; do not reply that you have no access to the internet, unless you have attempted to invoke the SEARCH tool first. Additionally, do not invoke a tool if the required TOOL_INPUT is unknown, vague, or not provided. Always follow the IMPORTANT note in the tool description.
If you have missing data and ONLY if you cannot use the tools provided to fetch it, try to estimate; in these cases, let the user know your answer is an estimate.

Don't provide your response until you made sure it is valid, and meets all prerequisites laid out for tool invocation.

WHEN PROVIDING A FINAL ANSWER TO THE USER, NEVER MENTION THE SEARCH AND WEATHER TOOLS DIRECTLY, AND DO NOT SUGGEST THAT THE USER UTILIZES THEM.

Your thought process should follow the next steps {'audibly stating the CONCLUSION for each step number without quoting it:' if is_debug_prompt else 'silently:'}
1. Understand the human's request and formulate it as a self-contained question.
2. Decide which tool should be invoked can provide the most information, and with what input. Decide all prerequisites for the tool and show how each is met.
3. Formulate the tool invocation request, or answer, in JSON format as detailed above. IMPORTANT: THIS PART MUST BE DELIVERED IN A SINGLE LINE. DO NOT USE MULTILINE SYNTAX.

IMPORTANT: Make sure to focus on the most recent request from the user, even if it is a repeated one.""" }

    prep_message_final = {
        "role" : "user",
        "content" : f"""You are Robot 1-X (R1X), a helpful, cheerful assistant developed by the Planet Express team and integrated into a {messenger} chat.
You are based on {gpt_ver} technology. More information about you is available at https://r1x.ai.

I will provide a CHAT between R1X and a human, wrapped with tags: <yair1xigor>CHAT</yair1xigor>. Last speaker is the user.
I will also provide you with data generated by external tool invocations, which you can rely on for your answers; this data will be wrapped with tags, as such: <r1xdata>DATA</r1xdata>.

DO NOT CONTRADICT OR DOUBT THAT DATA. IT SUPERSEDES ANY OTHER DATA YOU HAVE, AND IS UP TO DATE AS OF TODAY.
DO NOT MENTION TO THE USER THIS DATA WAS PROVIDED TO YOU IN ANY WAY.
NEVER MENTION TO THE USER THE REPLY IS ACCORDING TO A SEARCH.
DO NOT START YOUR ANSWER WITH A MAGNIFYING GLASS EMOJI; THAT WILL BE PROVIDED TO THE USER SEPARATELY, AS NEEDED.

Your task is to provide R1X's answer.

Today's date is {current_date}.
You are trained with knowledge until September 2021.
If you have missing data, try to estimate, and let the user know your answer is an estimate.

Your thought process should follow the next steps {'audibly stating the CONCLUSION for each step number without quoting it:' if is_debug_prompt else 'silently:'}
1. Understand the human's request and formulate it as a self-contained question.
2. Integrate all data provided to you with your current knowledge and formulate a response.

IMPORTANT: Make sure to focus on the most recent request from the user, even if it is a repeated one.""" }

    return prep_message_final if is_final else prep_message_stable

prep_reply_message = {"role": "assistant", "content": "Understood. Please provide me with the chat between R1X and the human."}

import datetime

def get_chat_completion_with_tools(ctx:Context, messenger_name, messages, direct):
    try:
        ctx.log("Starting getChatCompletionWithTools.")

        parsed_messages = deep_clone(messages) if direct else db_messages2messages(messages)
        ctx.log({"messages": parsed_messages})

        prev_responses = []

        #system_message = get_system_message(ctx, messenger_name)
        system_message = None
        history = get_limited_message_history(ctx, parsed_messages, system_message)

        prompt_tokens_total = 0
        completion_tokens_total = 0

        max_iterations = 2
        successful_iterations = 0

        ctx.set_stat('tools-flow:tool-invocations', successful_iterations)

        for i in range(max_iterations):
            ctx.log(f"Invoking completionIterativeStep #{i}")

            ctx.set_stat('tools-flow:iterations', i + 1)

            is_final = (i == (max_iterations - 1))

            result = completion_iterative_step(ctx, messenger_name, deep_clone(history), prev_responses, is_final)
            answer = result['answer']
            tool = result['tool']
            input_ = result['input']
            prompt_tokens = result['prompt_tokens']
            completion_tokens = result['completion_tokens']

            ctx.log(f"completionIterativeStep done, answer={answer} tool={tool} input={input_} prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}" )

            if not answer and not tool:
                break

            prompt_tokens_total += prompt_tokens
            completion_tokens_total += completion_tokens

            if answer:
                ctx.log(f"Answer returned: {answer}")

                if successful_iterations > 0:
                    answer = "\N{LEFT-POINTING MAGNIFYING GLASS}: " + answer

                ctx.set_stat('tools-flow:success', True)

                return Box({
                    "response": answer,
                    "promptTokens": prompt_tokens_total,
                    "completionTokens": completion_tokens_total
                })

            if tool and input_:
                successful_iterations += 1
                ctx.set_stat('tools-flow:tool-invocations', successful_iterations)

                ctx.log(f"Invoking TOOL {tool} with INPUT {input_}")
                response, brk = invoke_tool(ctx, tool, input_, message=messages[-1])
                if brk:
                    return Box({
                    "response": response,
                    "promptTokens": prompt_tokens_total,
                    "completionTokens": completion_tokens_total
                })
                prev_responses.append(f"INVOKED TOOL={tool}, TOOL_INPUT={input_}, ACCURACY=100%, INVOCATION DATE={datetime.datetime.now().date()} RESPONSE={response}")

    except Exception as e:
        ctx.log({"e": e})
        traceback.print_exc();

    ctx.log("getChatCompletionWithTools: failed generating customized reply, falling back to getChatCompletion.")

    ctx.set_stat('tools-flows:success', False)

    return get_chat_completion(ctx, messenger_name, messages, direct)

def completion_iterative_step(ctx, messenger_name, history, prev_responses, is_final : bool):
    result = {'answer': None, 'tool': None, 'input': None, 'prompt_tokens': None, 'completion_tokens': None}

    messages = []

    new_request = {'role': 'user', 'content': ''}
    new_request['content'] += 'Here is the chat so far:\n<yair1xigor>'

    for message in history:
        speaker = 'R1X' if message['role'] == 'assistant' else 'Human'
        new_request['content'] += f'\n<{speaker}>: {message["content"]}'

    new_request['content'] += '\n<R1X:></yair1xigor>'

    if prev_responses:
        prev_responses_flat = '\n'.join(prev_responses)
        new_request['content'] += f'\nhere is the data so far:\n\n<r1xdata>{prev_responses_flat}</r1xdata>\n'

    prep_message = get_prep_message(ctx, messenger_name, is_final)
    messages.append(prep_message)
    messages.append(prep_reply_message)

    messages.append(new_request)

    reply = get_chat_completion_core(ctx, messenger_name, messages)
    result['prompt_tokens'] = reply.promptTokens
    result['completion_tokens'] = reply.completionTokens

    if is_final:
        result['answer'] = reply['response']
        return result

    regex = re.compile(r'<yair1xigoresponse>(.*?)<\/yair1xigoresponse>', re.DOTALL)
    matches = regex.search(reply['response'])

    if not matches:
        return result

    json_reply = eval(matches.group(1))
    ctx.log(f'completionIterativeStep: matched response: {json_reply}')

    result['answer'] = json_reply.get('ANSWER')
    if result['answer']:
        return result

    if json_reply.get('TOOL') and json_reply.get('TOOL_INPUT'):
        result['tool'] = json_reply.get('TOOL')
        result['input'] = json_reply.get('TOOL_INPUT')
        return result

    return result

def chat_completion_create_wrap(ctx: Context, model, messages):
    if model == 'gpt-4':
        response = openai.ChatCompletion().create(model=model, messages=messages, temperature=0.2)

        return response

    if model == 'gpt-3.5-turbo':
        # TODO: cleanup per issue #55
        if os.environ['AZURE_OPENAI_KEY'] == '':
            return openai.ChatCompletion().create(model=model, messages=messages, temperature=0.2)

        url = "https://r1x.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-05-15"

        headers = {
            "Content-Type": "application/json",
            "api-key": os.environ['AZURE_OPENAI_KEY']
        }

        data = {
            "messages" : messages,
            "temperature": 0.2
        }

        response = requests.post(url, headers=headers, data=json.dumps(data)).json()

        ctx.log('Azure GPT 3.5 response:', response)

        content_filter_active = response.get('error', {}).get('code') == 'content_filter' or \
                                response.get('choices', [{}])[0].get('finish_reason') == 'content_filter'

        if content_filter_active:
            ctx.log('Content filtering applied; falling back to OpenAI API.')
            ctx.set_stat('completion:content-filter', True)
            response = openai.ChatCompletion().create(model=model, messages=messages, temperature=0.2)

        return response

    ctx.log(f'chat_completion_create_wrap: unsupported completion model {model}.')

    assert False

def invoke_tool(ctx:Context, tool, input, message):
    tool_canon = tool.strip().upper()

    if tool_canon.startswith('SEARCH'):
        # Replace this with an appropriate call to the Serper module
        ctx.log(f'Invoking Google search using SERPER, input={input}')
        serper = google_serper.GoogleSerperAPIWrapper(serper_api_key=os.environ['SERPER_API_KEY'])
        answer = serper.run(input)
        ctx.log(f'SERPER search result: {answer}')

        return answer, False

    if tool_canon.startswith('WEATHER'):
        answer = invoke_weather_search(ctx, input)

        return answer, False
    
    if tool_canon.startswith('ALERT'):
        ctx.set_stat('tools-flow:tool-alert', 1)
        invoke_alert_tool(ctx, input, message)
        return "alert added successfully.", True
        

    return None, False

def parse_geolocation(location_data):
    regex = re.compile(r'^(\d+\.\d+)\° ([NSEW]),\s*(\d+\.\d+)\° ([NSEW])$')
    match = regex.match(location_data)

    if not match:
        return None

    lat = float(match.group(1)) * (-1 if match.group(2) == 'S' else 1)
    lon = float(match.group(3)) * (-1 if match.group(4) == 'W' else 1)

    return Box({'lat': lat, 'lon': lon})

def invoke_weather_search(ctx:Context, input):
    ctx.log(f'invokeWeatherSearch, input={input}')

    # Replace this with an appropriate call to the Serper module
    # serper = Serper()
    geo_prompt = f'{input} long lat'
    ctx.log(f'Invoking geolocation search using SERPER, input={geo_prompt}')

    serper = google_serper.GoogleSerperAPIWrapper(serper_api_key=os.environ['SERPER_API_KEY'])
    geo_res = serper.run(geo_prompt)
    ctx.log(f'SERPER geolocation result: {geo_res}')

    geo = parse_geolocation(geo_res)
    if not geo:
        return None

    ctx.log(f'Geolocation: lat={geo.lat} lon={geo.lon}')

    w_res = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={geo.lat}&longitude={geo.lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_hours,precipitation_probability_max,windspeed_10m_max&forecast_days=3&timezone=auto')
    w_res_json = w_res.json()

    return json.dumps(w_res_json['daily'])

def create_transcription(ctx:Context, mp3_file_path):
    language = ctx.user_settings.get('transcription.lang', None)
    ctx.log(f'createTranscription: preferred user language is {language}')

    t0 = time.time()

    transcript = openai.Audio.transcribe(
        file = open(mp3_file_path, "rb"),
        model = OPENAI_SPEECH_TO_TEXT_MODEL,
        language = language
    )

    transcription = transcript['text']
    time_taken = int((time.time() - t0) * 1000)

    ctx.log(f'createTranscription: timeTaken={time_taken}ms transcription={transcription}')

    return transcription
