import openai

ft_model = 'davinci:ft-personal-2022-11-02-07-53-35'

def gpt_pred(qualified_text):
    res = openai.Completion.create(model=ft_model, prompt=qualified_text + '->', max_tokens=10, temperature=0)
    return res['choices'][0]['text']