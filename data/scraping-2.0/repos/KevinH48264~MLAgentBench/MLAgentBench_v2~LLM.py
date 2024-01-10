'''
This should be a lightweight class and only contain actions / functions for the following:
- List files (environment for actions, )
- Read file
- Write file (only in workspace)
- Search
- Think
- Final answer
- An OpenAI Assistant class will also be provided to take in a prompt and give an output after running through all the Actions it decides is necessary. 
- Note: Logging is automatically invoked whenever an Action is invoked, saving state of work_dir in log_dir'''

import os
from functools import partial
import tiktoken
import json
from .schema import TooLongPromptError, LLMError
import json

enc = tiktoken.get_encoding("cl100k_base")

try:
    from helm.common.authentication import Authentication
    from helm.common.request import Request, RequestResult
    from helm.proxy.accounts import Account
    from helm.proxy.services.remote_service import RemoteService
    # setup CRFM API
    auth = Authentication(api_key=open("crfm_api_key.txt").read().strip())
    service = RemoteService("https://crfm-models.stanford.edu")
    account: Account = service.get_account(auth)
except Exception as e:
    print(e)
    print("Could not load CRFM API key crfm_api_key.txt.")

try:   
    import anthropic
    # setup anthropic API key
    anthropic_client = anthropic.Anthropic(api_key=open("claude_api_key.txt").read().strip())
except Exception as e:
    print(e)
    print("Could not load anthropic API key claude_api_key.txt.")
    
try:
    import openai
    from openai import OpenAI
    # setup OpenAI API key
    openai.organization, openai.api_key  =  open("openai_api_key.txt").read().strip().split(":")
    openai_client = OpenAI(api_key=openai.api_key)
except Exception as e:
    print(e)
    print("Could not load OpenAI API key openai_api_key.txt.")


def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    """ Log the prompt and completion to a file."""
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}")
        num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")


def complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT], model="claude-v1", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, **kwargs):
    """ Call the Claude API to complete a prompt."""
    print("CLAUDE WAS CALLED!")

    ai_prompt = anthropic.AI_PROMPT
    if "ai_prompt" in kwargs is not None:
        ai_prompt = kwargs["ai_prompt"]

    try:
        rsp = anthropic_client.completions.create(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {ai_prompt}",
            stop_sequences=stop_sequences,
            model=model,
            temperature=temperature,
            max_tokens_to_sample=max_tokens_to_sample,
            **kwargs
        )
    except anthropic.APIStatusError as e:
        print(e)
        raise TooLongPromptError()
    except Exception as e:
        raise LLMError(e)

    completion = rsp.completion
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def get_embedding_crfm(text, model="openai/gpt-4-0314"):
    request = Request(model="openai/text-similarity-ada-001", prompt=text, embedding=True)
    request_result: RequestResult = service.make_request(auth, request)
    return request_result.embedding 
    
def complete_text_crfm(prompt=None, stop_sequences = None, model="openai/gpt-4-0314",  max_tokens_to_sample=2000, temperature = 0.5, log_file=None, messages = None, **kwargs):
    
    random = log_file
    if messages:
        request = Request(
                prompt=prompt, 
                messages=messages,
                model=model, 
                stop_sequences=stop_sequences,
                temperature = temperature,
                max_tokens = max_tokens_to_sample,
                random = random
            )
    else:
        print("model", model)
        print("max_tokens", max_tokens_to_sample)
        request = Request(
                prompt=prompt, 
                model=model, 
                stop_sequences=stop_sequences,
                temperature = temperature,
                max_tokens = max_tokens_to_sample,
                random = random
        )
    
    try:      
        request_result: RequestResult = service.make_request(auth, request)
    except Exception as e:
        # probably too long prompt
        print(e)
        raise TooLongPromptError()
    
    if request_result.success == False:
        print(request.error)
        raise LLMError(request.error)
    completion = request_result.completions[0].text
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion

# TODO: clean this function code up!
def complete_text_openai(prompt, system_prompt="You are a helpful assistant.", stop_sequences=[], model="gpt-3.5-turbo-1106", max_tokens_to_sample=2000, temperature=0.2, log_file=None, json_required=False, tools=None, available_functions=None, **kwargs):
    # print("\nOpenAI model: ", model, "\nPrompt: ", prompt, "\nPrompt length: ", len(prompt))
    """ Call the OpenAI API to complete a prompt."""

    if json_required and (model == "gpt-3.5-turbo-1106" or model == "gpt-4-1106-preview"):
        raw_request = {
            "model": model,
            "response_format": { "type": "json_object" },
            "temperature": temperature,
            "max_tokens": max_tokens_to_sample,
            "stop": stop_sequences or None,  # API doesn't like empty list
            **kwargs
        }
    elif tools:
        raw_request = {
            "model": model,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": max_tokens_to_sample,
            "stop": stop_sequences or None,  # API doesn't like empty list
            "temperature": temperature,
            **kwargs
        }
    else:
        raw_request = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens_to_sample,
            "stop": stop_sequences or None,  # API doesn't like empty list
            **kwargs
        }
    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        response = openai_client.chat.completions.create(**{"messages": messages,**raw_request})
        # print("RESPONSE: ", response)
        completion = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls

        # Ensure that the completion is JSON parsable. If it isn't, ask GPT to make it JSON parsable by doubling the max tokens.
        if json_required and (model == "gpt-3.5-turbo-1106" or model == "gpt-4-1106-preview"):
            try:
                completion_json = json.loads(completion)
                print("In complete_text_openai(), Completion JSON: ", completion_json)
            except:
                print("In complete_text_openai(), COMPLETION NOT IN JSON")
                convert_to_json_prompt = f'''Close this incomplete JSON so that it's in proper JSON format: {completion}'''
                raw_request = {
                    "model": model,
                    "response_format": { "type": "json_object" },
                    "temperature": temperature,
                    "max_tokens": max_tokens_to_sample*2,
                    "stop": stop_sequences or None,  # API doesn't like empty list
                    **kwargs
                }
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": convert_to_json_prompt}]
                response = openai_client.chat.completions.create(**{"messages": messages,**raw_request})
                completion = response.choices[0].message.content
                # print("NEW COMPLETION: ", completion)

        if tool_calls:
            messages.append(response.choices[0].message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
                print(f"Function calling required action: \nfunction_name: {function_name}, \ntool_function.arguments: {function_args}")
            # second_response = openai_client.chat.completions.create(
            #     model=model,
            #     messages=messages,
            # )  # get a new response from the model where it can see the function response
            return "Function calling complete!"
    else:
        response = openai.Completion.create(**{"prompt": prompt,**raw_request})
        completion = response["choices"][0]["text"]

    # if log_file:
    #     with open(log_file, "a", 1) as log_file:
    #         log_file.write(f"\nPrompt: {prompt}\n\nCompletion: {completion}\n")
    return completion

def complete_text(prompt, model, log_file=None, json=False, **kwargs):
    """ Complete text using the specified model with appropriate API. """
    print("COMPLETE TEXT")

    if model.startswith("claude"):
        # use anthropic API
        completion = complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT, "Observation:"], log_file=log_file, model=model, **kwargs)
    elif "/" in model:
        # use CRFM API since this specifies organization like "openai/..."
        completion = complete_text_crfm(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    else:
        # use OpenAI API
        completion = complete_text_openai(prompt, model=model, json_required=json, **kwargs)
    return completion

# specify fast models for summarization etc
FAST_MODEL = "gpt-3.5-turbo-1106"
def complete_text_fast(prompt, **kwargs):
    print("COMPLETE_TEXT_FAST")
    return complete_text(prompt = prompt, model = FAST_MODEL, temperature =0.01, **kwargs)
# complete_text_fast = partial(complete_text_openai, temperature= 0.01)