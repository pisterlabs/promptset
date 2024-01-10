import json
import openai

API_KEY = open("API KEY", "r").read()
openai.api_key = API_KEY


def gpt_classifier(message):
    function = [
        {
            "name": "classify_Email",
            "description": "gets the company and topic mentioned in the email's subject line",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "the company mentioned in the emails subject line"
                    },
                    "topic": {
                        "type": "string",
                        "description": "the topic mentioned in the emails subject line"
                    }
                }
            }
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a email analyser. You are given an email and reliably filter out information within it."
            },
            {
                "role": "user",
                "content": f"Here is an email: {message}"
                           f"From just the subject line, extract the company talked about and the topic talked about."
                           f"Limit you analysis to just the text of subject line, do not read the body of the email."
                           f"The subject line you have to analyse is usually formatted like this:"
                           f"'Subject:' subject line text"
                           f"Provide just the answers, no other text"
            }
        ],
        functions=function,
        function_call={"name": "classify_Email"}
    )

    result = response['choices'][0]['message']['function_call']['arguments']
    json_result = json.loads(result)

    tokens = response['usage']['total_tokens']

    # print(json_result)

    return json_result["company"], json_result["topic"], tokens


def gpt_extractorNew(message):
    function = [
        {
            "name": "email_summarizer",
            "description": "give a summary of each relevant point and technical aspect talked about in the email",
            "parameters": {
                "type": "object",
                "properties": {
                    "summarization": {
                        "type": "string",
                        "description":  "returns a bullet point list of the main talking points and technical specifications from the email."
                                        "split into general talking points and technical specifications."
                                        "follows this schema:"
                                        ""
                                        "Talking points:"
                                        "- person x says ..."
                                        "- person x says ..."
                                        "- etc."
                                        ""
                                        "Technical Specifications:"
                                        "- Specification x"
                                        "- Specification y"
                                        "- etc."
                    }
                },
                "required": ["summarization"]
            }
        }
    ]

    Messages = [
        {
            "role": "system",
            "content": "You are a email summarizer."
        },
        {
            "role": "user",
            "content": f"You are an email summarizer. Here is an email:"
                       f"{message}."
                       f"Summarize and return all talking points from the emails body as a bullet point list."
                       f"Follow this schema:"
                       f""
                       f"Talking points:"
                       f"- person x says ..."
                       f"- person x says ..."
                       f"- etc."
                       f""
                       f"Technical Specifications:"
                       f"- Specification x"
                       f"- Specification y"
                       f"- etc."
                       f""
                       f"Disregard formalities and signatures."
                       f"If no talking point or specification can be found, return none in the schema."
                       f"Return just the list, no other text."
        },
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=Messages,
        functions=function,
        function_call={"name": "email_summarizer"}
    )

    result = response['choices'][0]['message']['function_call']['arguments']
    # print(response)
    json_result = json.loads(result, strict=False)

    # print(json_result)

    tokens = response['usage']['total_tokens']

    # print(json_result)

    return json_result["summarization"], tokens


def gpt_extractorAdd(message, context):
    function = [
        {
            "name": "email_summarizer",
            "description": "give a summary of each relevant point and technical aspect talked about in the email and add it to the context of previous summarizations",
            "parameters": {
                "type": "object",
                "properties": {
                    "summarization": {
                        "type": "string",
                        "description":  "a bullet point list of the main talking points and technical specifications from the email."
                                        "split into general talking points and technical specifications."
                                        ""
                                        "follows this schema:"
                                        ""
                                        "Talking points:"
                                        "- person x says ..."
                                        "- person x says ..."
                                        "- etc."
                                        ""
                                        "Technical Specifications:"
                                        "- Specification x"
                                        "- Specification y"
                                        "- etc."
                    }
                },
                "required": ["summarization"]
            }
        }
    ]

    Messages = [
        {
            "role": "assistant",
            "content": f"The previous emails summarization:"
                       f"{context}"
        },
        {
            "role": "system",
            "content": "You are a email summarizer."
        },
        {
            "role": "user",
            "content": f"You are an email summarizer. Here is an email:"
                       f"{message}."
                       f"Taking the previous responses as context, summarize and return all talking points from the emails body as a bullet point list."
                       f"If information is already in the context, add the context information to the information from the email."
                       f"Follow this bullet point list schema:"
                       f""
                       f"Talking points:"
                       f"- person x says ..."
                       f"- person x says ..."
                       f"- etc."
                       f""
                       f"Technical Specifications:"
                       f"- Specification x"
                       f"- Specification y"
                       f"- etc."
                       f""
                       f"Disregard formalities and signatures."
                       f"Return just the list, no other text."
        },
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=Messages,
        functions=function,
        function_call={"name": "email_summarizer"}
    )

    result = response['choices'][0]['message']['function_call']['arguments']
    # print(response)
    json_result = json.loads(result, strict=False)

    # print(json_result)

    tokens = response['usage']['total_tokens']

    # print(json_result)

    return json_result["summarization"], tokens


def gpt_entryComparer(company1, topic1, company2, topic2):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You compare two different email subject lines and return whether they belong to the same email chain or not"
            },
            {
                "role": "user",
                "content": f"Below are the email subject company and object from two emails:"
                           f"Email 1: {company1} - {topic1}"
                           f"Email 2: {company2} - {topic2}"
                           f"Do these emails regard the sam company and product?"
                           f"Return just TRUE or FALSE as your answer"
            }
        ],
    )

    result = response['choices'][0]['message']['content']

    tokens = response['usage']['total_tokens']

    return result, tokens
