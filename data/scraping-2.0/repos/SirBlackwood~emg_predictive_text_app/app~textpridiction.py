import openai
import re

openai.api_key="" #Here is your OpenAi api-key 
model_engine='text-davinci-003'


def predict_text(two_letters, context=None, word_num=0):
    if not context:
        prompt = f'Я подаю тебе две буквы: "{two_letters}". Выдай мне 9 самых частых в использовании слов, ' \
               f'у которых первая буква "{two_letters[0]}", а вторая "{two_letters[1]}" и ' \
               f'вставь каждое слово в фигурные скобки "{{}}". ' \
               f'Кроме этих слов ничего не пиши.'
    else:
        prompt = f'Вот начало предложения "{context}". Как бы ты продолжил его, если следующее слово начинается с "{two_letters}" (это слово не должно повторяться в предложениях). ' \
                 f'Выведи девять вариантов этого предложения целиком и возьми их в фигурные скобки ({{}}). Кроме этих предложений больше ничего не пиши и не используй знаки препинания'
    completion = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=1024, n=1, stop=None,
                                        temperature=0.7)
    message = completion.choices[0].text
    print(message)
    pattern = r"\{([^{}]+)\}"
    if not context:
        return re.findall(pattern, message)
    else:
        messages = re.findall(pattern, message)
        words = []
        for m in messages:
            words.append(re.sub(r'[^\w\s]', '', m).split()[word_num])
        return words
