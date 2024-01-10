from scripts.physton_prompt.get_lang import get_lang
from scripts.physton_prompt.get_translate_apis import unprotected_translate_api_config


def gen_openai(messages, api_config):
    api_config = unprotected_translate_api_config('chatgpt_key', api_config)
    if api_config.get('api_base', 'https://api.openai.com/v1') == "g4f":
        import g4f
        messages[1]["content"] = "「" + messages[1]["content"] + "」を30要素でプロンプトを生成してください。"
        completion = g4f.ChatCompletion.create(
            model="gpt-3.5-turbo",
            provider=g4f.Provider.DeepAi,
            messages=messages,
            stream=False
        )
        if "「" in completion:
            completion = completion.split("「")[1].split("」")[0]
        completion = completion.replace(".", "")
        return completion
    else:
        import openai
        openai.api_base = api_config.get('api_base', 'https://api.openai.com/v1')
        openai.api_key = api_config.get('api_key', '')
        model = api_config.get('model', 'gpt-3.5-turbo')
        if not openai.api_key:
            raise Exception(get_lang('is_required', {'0': 'API Key'}))
        if not messages or len(messages) == 0:
            raise Exception(get_lang('is_required', {'0': 'messages'}))
        completion = openai.ChatCompletion.create(model=model, messages=messages, timeout=60)
        if len(completion.choices) == 0:
            raise Exception(get_lang('no_response_from', {'0': 'OpenAI'}))
        content = completion.choices[0].message.content
        return content
