import openai
import requests

# AWS Secret Manager
AWS_SECRET_NAME = "OpenAI/CareDaily"

# ID of the model to use. You can use the [List models API](https://platform.openai.com/docs/api-reference/models/list) to see all of your available models, or see our [Model](https://platform.openai.com/docs/models/overview) overview for descriptions of them.
DEFAULT_COMPLETION_MODEL = "text-davinci-003"
DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"

# What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
DEFAULT_TEMPERATURE = 0.9

# The maximum number of tokens to generate in the completion.
DEFAULT_MAX_TOKEN = 300

# An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
DEFAULT_TOP_P = 1

# Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
DEFAULT_FREQUENCY_PENALTY = 0.2

# Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
DEFAULT_PRESENCE_PENALTY = 0

# Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
STOP_SEQUENCE_INPUT         = "USER:"
STOP_SEQUENCE_OUTPUT        = "SERVICE:"
DEFAULT_STOP_SEQUENCES = [
    STOP_SEQUENCE_INPUT,
    STOP_SEQUENCE_OUTPUT,
]

def _get_api_key(botengine):
    """
    Get the API key from the AWS Secret Manager
    :param botengine: BotEngine environment
    :return: API key
    """
    secret = botengine.get_secret(secret_name=AWS_SECRET_NAME)
    if secret is None:
        botengine.get_logger().warning("genai: Failed to authenticate. Missing authentication.")
        return None
    botengine.get_logger().debug("genai: Secret: {}".format(secret))
    import json
    secret = json.loads(secret)
    if "appname" not in secret or "appsecret" not in secret:
        botengine.get_logger().warning("genai: Failed to authenticate. Missing authentication.")
        return None
    return secret["appsecret"]

def open_ai_response(botengine, location_object, content):
    """
    Generate a resposne from OpenAI
    :param botengine: BotEngine environment
    :param location_object: Location object
    :param content: Content to send to OpenAI
    :return: Response from OpenAI

    Example content:
    {
        "prompt": "Hello",
        "openai_params": {
            temperature: 0.9,
            max_token: 150,
            top_p: 1,
            frequency_penalty: 0.2,
            presence_penalty: 0,
        },
        "model": 'gpt-3.5-turbo',
        "endpoint": 'https://api.openai.com/v1/chat/completions'
    }
    """
    api_key = _get_api_key(botengine)
    if api_key is None:
        # TODO: Clean up this error handling
        botengine.get_logger().warning("genai.open_ai_response(): Missing API Key")
        return None
    
    if content is None or 'prompt' not in content or content['prompt'] is None:
        botengine.get_logger().warning("genai.open_ai_response(): No prompt provided")
        return None

    # Set up the model and prompt
    model_engine = DEFAULT_COMPLETION_MODEL
    temperature = DEFAULT_TEMPERATURE
    max_token = DEFAULT_MAX_TOKEN
    top_p = DEFAULT_TOP_P
    frequency_penalty = DEFAULT_FREQUENCY_PENALTY
    presence_penalty = DEFAULT_PRESENCE_PENALTY
    stop = DEFAULT_STOP_SEQUENCES

    # Override the model and prompt if they're provided
    if 'model' in content and content['model'] is not None:
        model_engine = content['model']
    if 'openai_params' in content and content['openai_params'] is not None:
        openai_params = content['openai_params']
        if 'temperature' in openai_params and openai_params['temperature'] is not None:
            temperature = openai_params['temperature']
        if 'max_token' in openai_params and openai_params['max_token'] is not None:
            max_token = openai_params['max_token']
        if 'top_p' in openai_params and openai_params['top_p'] is not None:
            top_p = openai_params['top_p']
        if 'frequency_penalty' in openai_params and openai_params['frequency_penalty'] is not None:
            frequency_penalty = openai_params['frequency_penalty']
        if 'presence_penalty' in openai_params and openai_params['presence_penalty'] is not None:
            presence_penalty = openai_params['presence_penalty']
        if 'stop' in openai_params and openai_params['stop'] is not None:
            stop = openai_params['stop']

    # Set up the OpenAI API client
    openai.api_key = api_key

    # Generate a response
    try:
        import json

        botengine.get_logger().info("genai.open_ai_response(): Content: {}".format(json.dumps(content)))
        
        import timeit
        timestamp = timeit.default_timer()
        completion = openai.Completion.create(engine=model_engine, prompt=content['prompt'], temperature=temperature,
                                            max_tokens=max_token, top_p=top_p, frequency_penalty=frequency_penalty,
                                            presence_penalty=presence_penalty, stop=stop)
        try:
            # Track everything but the response and the prompt to limit the amount of data we're storing
            # print("genai.open_ai_response(): Completion: {}".format(completion.__dict__))
            """
            {
                "_previous": {
                    "choices": [{"finish_reason": "stop", "index": 0, "logprobs": null, "text": "Hi"}], 
                    "created": 1688061227, 
                    "id": "cmpl-7Wq7vCLbCAInd6dPPJBUxiehdmw1G", 
                    "model": "text-davinci-003", 
                    "object": "text_completion", 
                    "usage": {"completion_tokens": 1, "prompt_tokens": 6, "total_tokens": 7}}, 
                    "_response_ms": 293, 
                    "_retrieve_params": {}, 
                    "api_base_override": null, 
                    "api_key": "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", 
                    "api_type": null, 
                    "api_version": null, 
                    "engine": "text-davinci-003", 
                    "organization": "care-daily-xf7fyx", 
                    "test": true}
            """
            import signals.analytics as analytics
            properties = completion.__dict__["_previous"]
            analytics.track(botengine, location_object, "genai_completion", properties={
                "created" : completion.created,
                "model" : completion.model,
                "object" : completion.object,
                "usage" : completion.usage,
                "organization": completion.organization,
                "response_time_ms": int((timeit.default_timer() - timestamp) * 1000),
            })
        except:
            botengine.get_logger().warning("genai.open_ai_response(): Error tracking completion analytics: {}".format(e))
            pass
        botengine.get_logger().info("genai.open_ai_response(): Completion: {}".format(json.dumps(completion)))

        response = completion.choices[0]['text']
        botengine.get_logger().info("genai.open_ai_response(): Open AI Response: {}".format(response))
        return response
    except Exception as e:
        botengine.get_logger().error("genai.open_ai_response(): Error generating response: {}".format(e))
        return _("I'm sorry, I don't know how to respond to that.")

def open_ai_image(botengine, prompt):
    # Make the request to the OpenAI API
    resp = requests.post(
        'https://api.openai.com/v1/images/generations',
        headers={'Authorization': f'Bearer {API_KEY}'},
        json={'prompt': prompt, 'n': 1, 'size': '1024x1024'},
        timeout=10
    )
    response_text = json.loads(resp.text)

    return response_text['data'][0]['url']

def open_ai_response_expert(botengine, content):
    """
    Example content:

    content = {
            "prompt": prompt,
            "openai_params": None,
            "model": 'gpt-3.5-turbo',
            "endpoint": 'https://api.openai.com/v1/chat/completions'
        }
    """

    temperature = None
    max_token = None
    top_p1 = None
    frequency_penalty = None
    presence_penalty = None

    model = None
    api_key = None
    prompt = None


    openai_params = content['openai_params']
    prompt = content['prompt']
    endpoint = content['endpoint']
    model = content['model']

    if prompt is None or endpoint is None:
        return

    if openai_params is None:
        temperature = 0.9
        max_token = 150
        top_p1 = 1
        frequency_penalty = 0.2
        presence_penalty = 0
    else:
        temperature = openai_params[0]
        max_token = openai_params[1]
        top_p1 = openai_params[2]
        frequency_penalty = openai_params[3]
        presence_penalty = openai_params[4]

    data = {'model': model,
            'messages': [{"role": "user", "content": prompt}],
            'temperature': temperature,
            'max_tokens': max_token,
            'top_p': top_p1,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty}

    headers = {'Authorization': f'Bearer {API_KEY}'}
    # Make the request to the OpenAI API
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        botengine.get_logger().warning("genai.open_ai(): Error: {}".format(err))
        return None
    
    try:
        from json import JSONDecodeError
        result = response.json()
    except JSONDecodeError as err:
        botengine.get_logger().warning("genai.open_ai(): Error: {}".format(err))
        return None


    final_result = ''
    for i in range(0, len(result['choices'])):
        final_result+=result['choices'][i]['message']['content']

    botengine.get_logger().info("genai.open_ai(): Open AI Response: {}".format(final_result))
    return final_result