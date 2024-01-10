from flask import make_response

def set_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

def handle_request(request):
    if request.method == 'OPTIONS':
        # preflight request
        response = make_response()
        return set_cors_headers(response)

    elif request.method == 'POST':
        return upload(request)

    else:
        response = make_response('Method not supported')
        response.status_code = 405
        return set_cors_headers(response)

def upload(request):
    import os
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    # initialize the OpenAI model
    chat = ChatOpenAI(temperature=0, openai_api_key = os.getenv('OPENAI_API_KEY'))

    # create the prompt templates
    system_template = """You are a helpful assistant that translates {input_language} to {output_language}.
        The content you translate are srt files. Do not attempt to translate it word by word. Rather, be logical and translate the meaning of the text, keeping the time stamps in place.
        Finally, do not include anything other the translated srt file in your response."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
 
    file = request.files['file']

    if not file:
        response = make_response('No file part')
        response.status_code = 400
        return set_cors_headers(response)


    if file.filename == '':
        response = make_response('No selected file')
        response.status_code = 400
        return set_cors_headers(response)

    file_contents = file.read().decode('utf-8')

    # translate the contents
    response = chat(
        chat_prompt.format_prompt(
            input_language="English", output_language="Finnish", text=file_contents
        ).to_messages()
    )
    translated_contents = response.content
    response = make_response(translated_contents)

    # return translated contents
    return set_cors_headers(response)
    

