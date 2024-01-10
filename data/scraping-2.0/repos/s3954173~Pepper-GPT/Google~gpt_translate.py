import functions_framework
import openai

@functions_framework.http
def gpt_translate(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    openai.api.key = request_json['openai_api_key']
    phrase = request_json['phrase']
    language = request_json['language']
    
    model_engine = "test-davinci-003"
    prompt = (f"Translate {phrase} into {language}.")

    completions = openai.Completion.create(
             engine=model_engine,
             prompt=prompt,
             max_tokens=1024,
             n=1,
             stop=None,
             temperature=0.5,
             )

    message = completions.choices[0].text
    return message.strip()

