import os
import openai

class ClassicGPTResponseGetter:
    '''
    === Classic GPT: No function calling version
    '''
    # Get gpt response
    def get_gpt_response_classic(self, messages: list, model_name='gpt-4', temp=1.0):
        openai.organization = os.environ['OPENAI_KUNLP']
        openai.api_key = os.environ['OPENAI_API_KEY_TIMELINE']

        response = openai.ChatCompletion.create(
            model=model_name,
            temperature=temp,
            messages=messages,
            request_timeout=180
        )

        response_message = response['choices'][0]['message']
        assistant_message = {'role': 'assistant', 'content': response_message['content']}
        messages.append(assistant_message)

        print('No function calling (classic_gpt.py).')
        return messages

