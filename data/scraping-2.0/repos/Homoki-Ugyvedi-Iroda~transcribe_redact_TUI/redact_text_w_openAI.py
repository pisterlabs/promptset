import os
import openai
import json

error_mapping = {
    openai.error.AuthenticationError: "The service is not accessible, please try again later! [Authentication Error]",
    openai.error.RateLimitError: "Too many requests sent by the demo or server overloaded, please try again later! [RateLimit Error]",
    openai.error.InvalidRequestError: "There was a problem with the format of the request sent. We might receive this error also when the OpenAI API is overloaded. Please try again later! [InvalidRequest Error]",
    openai.error.APIConnectionError: "There was a problem with accessing the OpenAI API. Please try again later! [APIConnectionError]",
    openai.error.ServiceUnavailableError: "There was a problem with accessing the OpenAI API (service unavailable). Please try again later! [ServiceUnavailable Error]",
    openai.error.Timeout: "The response has not been received in time (timeout). Please try again later! [Timeout]",
    openai.error.OpenAIError: "Some general error with OpenAI happened. [InvalidRequest Error: 400]",
    Exception: "Some general error not related to OpenAI happened."
}

class OpenAIRedactor:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def text_to_json(self, filename) -> dict:
        if os.path.getsize(filename) == 0:
            return {}
        with open(filename, encoding='utf-8', mode='r', errors="ignore") as file:
            text_data = json.load(file)
        return text_data

    def read_prompt_instructions(self) -> str:
        with open(os.path.join(os.getcwd(), 'static', 'prompt_instructions.txt'), encoding='utf8', mode='r', errors="ignore") as f:
            prompt_instructions = f.read()
        return prompt_instructions

    def read_prompt_qa_examples(self) -> dict:
        return self.text_to_json(
            os.path.join(os.getcwd(), 'static', 'prompt_qa_examples.json'))    

    def construct_prompt_chat_gpt(self, user_input, system_prompt, model_config, read_prompt = False, read_qa_examples = False):
        prompt_instructions = ""
        prompt_qa_examples = ""
        if model_config == "gpt-3.5-turbo": #gpt-3.5 tends to ignore system_prompt [shall verify if changed w/ later versions]
            user_input = system_prompt + ". " + user_input
            system_prompt = ""
        if read_prompt:
            prompt_instructions = self.read_prompt_instructions().strip()
        if read_qa_examples:
            prompt_qa_examples = self.read_prompt_qa_examples()
        messages = [{
            "role": "system",
            "content": system_prompt
            }]
        #size_of_messages = util.get_token_length(json.dumps(messages))
        if len(prompt_qa_examples) > 0:           
            if len(prompt_instructions) > 0:            
                messages.append({
                    "role": "user",
                    "content": prompt_instructions + '\n\n' + prompt_qa_examples[0]["q"]
                    },
                    {
                    "role": "assistant",
                    "content": prompt_qa_examples[0]["a"]
                    })
                for i in range(1, len(prompt_qa_examples)):
                    messages.append({
                        "role": "user",
                        "content": prompt_qa_examples[i]["q"]
                        },
                        {
                        "role": "assistant",
                        "content": prompt_qa_examples[i]["a"]
                        })
            else:
                for example in prompt_qa_examples:
                    messages.append({
                        "role": "user",
                        "content": example["q"]
                        },
                        {
                        "role": "assistant",
                        "content": example["a"]
                        })
        else:
            if len(prompt_instructions) > 0:        
                messages.append({
                    "role": "user",
                    "content": prompt_instructions
                    })
        messages.append({
            "role": "user",
            "content": user_input
            })
        return messages


    def call_openAi_redact(self, user_input: str, system_prompt: str, model_config: str = "gpt-3.5-turbo", max_completion_length: int = 2048, timeout: int = "") -> str:
        messages = self.construct_prompt_chat_gpt(user_input, system_prompt, model_config, read_qa_examples=False, read_prompt=False)
        #size_of_messages = util.get_token_length(json.dumps(messages))
        
        try:
                completion = openai.ChatCompletion.create(
                        model=model_config,
                        max_tokens=max_completion_length,
                        messages=messages,
                        temperature=0.0,
                        request_timeout=timeout,
                )
                response_toolbot = completion['choices'][0]['message']['content']

                return response_toolbot

        except tuple(error_mapping.keys()) as error:
            return f"[Error]: {error}, {error_mapping[type(error)]}, \n See details at https://platform.openai.com/docs/guides/error-codes/python-library-error-types"
    