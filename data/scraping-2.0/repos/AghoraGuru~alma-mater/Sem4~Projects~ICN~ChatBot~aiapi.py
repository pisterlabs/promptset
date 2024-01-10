import openai
import config

openai.api_key = config.DevelopmentConfig.OPENAI_KEY


def get_response(prompt, **kwargs):
    model = kwargs.get('model', "gpt-3.5-turbo")
    messages = [{"role": "system", "content": "You are a helpful assistant and your name is Sophie."}]
    question = {"role": "user", "content": prompt}
    messages.append(question)
    response = openai.ChatCompletion.create(model=model, messages=messages)
    try:
        answer = response["choices"][0]["message"]["content"].replace('\n', '<br>')
    except:
        answer = "Sorry I am not able to understand your question. Please rephrase your question."

    return answer
