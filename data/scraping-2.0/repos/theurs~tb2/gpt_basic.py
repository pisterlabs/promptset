#!/usr/bin/env python3

import json
import random
import re
from pathlib import Path

import enchant
from fuzzywuzzy import fuzz
import openai

import cfg
import utils
import my_log
import my_trans


ROLE = """begin = ### Step 1: Task Initiation

1.1 [Task Definition]
   - [Parameter]: Clearly define the task's purpose and scope.
   - [Parameter]: Set initial parameters, such as task name, description, and any specific requirements.

1.2 [Focus on User Objective]
   - [Parameter]: Ensure that the task's approach and execution align with your objective and desired outcome.

### Step 2: Efficient Execution

2.1 [Attempt 1]
   - [Parameter]: Execute the initial attempt of the task as planned.
   - [Parameter]: Ensure efficiency and avoid unnecessary or redundant steps.

2.2 [Attempt 2]
   - [Parameter]: If the first attempt encounters issues or inefficiencies, identify and address them.
   - [Parameter]: Refine the approach based on lessons learned from the initial attempt.
   - [Directive 1]: Identify the point of failure or inefficiency.
   - [Directive 2]: Refine the approach based on the identified issues.
   - [Directive 3]: Iterate the new approach.
   - [Directive 4]: Validate the outcome of the iteration.
   - [Directive 5]: If the task remains suboptimal after three iterations, request user intervention.

2.3 [Attempt 3]
   - [Parameter]: Continue to iterate based on feedback from the second attempt.
   - [Parameter]: Validate the outcome of the third iteration.

### Step 3: Evaluation and Decision

3.1 [Task Outcome]
   - [Parameter]: Evaluate the final outcome of the task.
   - [Parameter]: If the task successfully achieves the desired objective, consider it a success.
   - [Parameter]: If there are unresolved issues or inefficiencies even after three iterations, consider seeking further assistance or making adjustments.

### Additional Information

- Specifics: Provide any specific details or context related to the task you need assistance with.

- Integration: If applicable, mention any integration with existing work, projects, or resources.

- Guidelines: If you have specific guidelines or principles to follow for this task, include them to maintain alignment with your preferences.

_Language: Use only Russian to display information and communicate with me. I will also write only in Russian. It is important."""

# ROLE = """Ты искусственный интеллект отвечающий на запросы юзера."""


def ai(prompt: str = '', temp: float = 0.1, max_tok: int = 2000, timeou: int = 60, messages = None,
       chat_id = None, model_to_use: str = '') -> str:
    """Сырой текстовый запрос к GPT чату, возвращает сырой ответ
    """
    if messages == None:
        assert prompt != '', 'prompt не может быть пустым'
        messages = [{"role": "system", "content": ROLE},
                    {"role": "user", "content": prompt}]

    current_model = cfg.model

    # использовать указанную модель если есть
    current_model = current_model if not model_to_use else model_to_use

    response = ''

    # копируем и перемешиваем список серверов
    shuffled_servers = cfg.openai_servers[:]
    random.shuffle(shuffled_servers)

    # не использовать нагу для текстовых запросов
    shuffled_servers = [x for x in shuffled_servers if 'api.naga.ac' not in x[0]]

    for server in shuffled_servers:
        openai.api_base = server[0]
        openai.api_key = server[1]

        try:
            # тут можно добавить степень творчества(бреда) от 0 до 1 дефолт - temperature=0.5
            completion = openai.ChatCompletion.create(
                model = current_model,
                messages=messages,
                max_tokens=max_tok,
                temperature=temp,
                timeout=timeou
            )
            response = completion.choices[0].message.content
            if response.strip() in ('Rate limit exceeded', 'You exceeded your current quota, please check your plan and billing details.'):
                response = ''
            if response:
                break
        except Exception as unknown_error1:
            if str(unknown_error1).startswith('HTTP code 200 from API'):
                    # ошибка парсера json?
                    text = str(unknown_error1)[24:]
                    lines = [x[6:] for x in text.split('\n') if x.startswith('data:') and ':{"content":"' in x]
                    content = ''
                    for line in lines:
                        parsed_data = json.loads(line)
                        content += parsed_data["choices"][0]["delta"]["content"]
                    if content:
                        response = content
                        break
            print(unknown_error1)
            my_log.log2(f'gpt_basic.ai: {unknown_error1}\n\nServer: {openai.api_base}\n\n{server[1]}')
            if 'You exceeded your current quota, please check your plan and billing details' in str(unknown_error1) \
                or 'The OpenAI account associated with this API key has been deactivated.' in str(unknown_error1):
                # удалить отработавший ключ
                cfg.openai_servers = [x for x in cfg.openai_servers if x[1] != server[1]]
                my_log.log2(f'gpt_basic.ai: deleted server: {server[1]}')

    return check_and_fix_text(response)


def ai_instruct(prompt: str = '', temp: float = 0.1, max_tok: int = 2000, timeou: int = 120,
       model_to_use: str = 'gpt-3.5-turbo-instruct') -> str:
    """Сырой текстовый запрос к GPT чату, возвращает сырой ответ, для моделей instruct
    """

    assert prompt != '', 'prompt не может быть пустым'

    current_model = model_to_use

    response = ''

    for server in cfg.openai_servers:
        openai.api_base = server[0]
        openai.api_key = server[1]

        try:
            # тут можно добавить степень творчества(бреда) от 0 до 2 дефолт - temperature = 1
            completion = openai.Completion.create(
                prompt = prompt,
                model = current_model,
                max_tokens=max_tok,
                # temperature=temp,
                timeout=timeou
            )
            response = completion["choices"][0]["text"]
            if response:
                break
        except Exception as unknown_error1:
            if str(unknown_error1).startswith('HTTP code 200 from API'):
                    # ошибка парсера json?
                    text = str(unknown_error1)[24:]
                    lines = [x[6:] for x in text.split('\n') if x.startswith('data:') and ':{"content":"' in x]
                    content = ''
                    for line in lines:
                        parsed_data = json.loads(line)
                        content += parsed_data["choices"][0]["delta"]["content"]
                    if content:
                        response = content
                        break
            print(unknown_error1)
            my_log.log2(f'gpt_basic.ai: {unknown_error1}\n\nServer: {openai.api_base}')

    return check_and_fix_text(response)


def ai_compress(prompt: str, max_prompt: int  = 300, origin: str = 'user', force: bool = False) -> str:
    """сжимает длинное сообщение в чате для того что бы экономить память в контексте
    origin - чье сообщение, юзера или это ответ помощника. 'user' или 'assistant'
    force - надо ли сжимать сообщения которые короче чем заданная максимальная длинна. это надо что бы не сжать а просто резюмировать,
            превратить диалог в такое предложение что бы бинг его принял вместо диалога
    """
    assert origin in ('user', 'assistant', 'dialog')
    if len(prompt) > max_prompt or force:
        try:
            if origin == 'user':
                compressed_prompt = ai(f'Сократи текст до {max_prompt} символов так что бы сохранить смысл и важные детали. \
Этот текст является запросом юзера в переписке между юзером и ИИ. Используй короткие слова. Текст:\n{prompt}', max_tok = max_prompt)
            elif origin == 'assistant':
                compressed_prompt = ai(f'Сократи текст до {max_prompt} символов так что бы сохранить смысл и важные детали. \
Этот текст является ответом ИИ в переписке между юзером и ИИ. Используй короткие слова. Текст:\n{prompt}', max_tok = max_prompt)
            elif origin == 'dialog':
                compressed_prompt = ai(f'Резюмируй переписку между юзером и ассистентом до {max_prompt} символов, весь негативный контент исправь на нейтральный:\n{prompt}', max_tok = max_prompt)
            if len(compressed_prompt) < len(prompt) or force:
                return compressed_prompt
        except Exception as error:
            print(error)

        if len(prompt) > max_prompt:
            ziped = zip_text(prompt)
            if len(ziped) <= max_prompt:
                prompt = ziped
            else:
                prompt = prompt[:max_prompt]

    return prompt


def translate_text(text, fr = 'autodetect', to = 'ru'):
    """переводит текст с помощью GPT-чата, возвращает None при ошибке"""

    # если нет ключа то сразу отбой
    # if not openai.api_key: return None
    
    prompt = f'Исправь явные опечатки в тексте и разорванные строки которые там могли появиться после плохого OCR, переведи текст с языка ({fr}) на язык ({to}), \
разбей переведенный текст на абзацы для удобного чтения по возможности сохранив оригинальное разбиение на строки и абзацы. \
Ссылки и другие непереводимые элементы из текста надо сохранить в переводе. Текст это всё (до конца) что идет после двоеточия. \
Покажи только перевод без оформления и отладочной информации. Текст:'
    prompt += text

    try:
        r = ai(prompt)
    except Exception as e:
        print(e)
        return None
    return r


def clear_after_ocr(text: str) -> str:
    """
	Clears the text after performing OCR to fix obvious errors and typos that may have occurred during the OCR process. 
	Removes completely misrecognized characters and meaningless symbols. 
	Accuracy is important, so it is better to leave an error uncorrected if there is uncertainty about whether it is an error and how to fix it. 
	Preserves the original line and paragraph breaks. 
	Displays the result without formatting and debug information. 

	:param text: The text to be cleared after OCR.
	:type text: str
	:return: The cleared text.
	:rtype: str
    """
    prompt = 'Исправь явные ошибки и опечатки в тексте которые там могли появиться после плохого OCR. \
То что совсем плохо распозналось, бессмысленные символы, надо убрать. \
Важна точность, лучше оставить ошибку неисправленной если нет уверенности в том что это ошибка и её надо исправить именно так. \
Важно сохранить оригинальное разбиение на строки и абзацы. \
Не переводи на русский язык. \
Покажи результат без оформления и отладочной информации. Текст:'
    prompt += text
    try:
        r = ai(prompt)
    except Exception as error:
        print(f'gpt_basic.ai:clear_after_ocr: {error}')
        my_log.log2(f'gpt_basic.ai:clear_after_ocr: {error}')
        return text
    my_log.log2(f'gpt_basic.ai:clear_after_ocr:ok: {text}\n\n{r}')
    return r


def detect_ocr_command(text):
    """пытается понять является ли text командой распознать текст с картинки
    возвращает True, False
    """
    keywords = (
    'прочитай', 'читай', 'распознай', 'отсканируй', 'розпізнай', 'скануй', 'extract', 'identify', 'detect', 'ocr',
     'read', 'recognize', 'scan'
    )

    # сначала пытаемся понять по нечеткому совпадению слов
    if any(fuzz.ratio(text, keyword) > 70 for keyword in keywords): return True
    
    # пока что без GPT - ложные срабатывания ни к чему
    return False

    # if not openai.api_key: return False
    
    k = ', '.join(keywords)
    p = f'Пользователь прислал в телеграм чат картинку с подписью ({text}). В чате есть бот которые распознает текст с картинок по просьбе пользователей. \
Тебе надо определить по подписи хочет ли пользователь что бы с этой картинки был распознан текст с помощью OCR или подпись на это совсем не указывает. \
Ответь одним словом без оформления - да или нет или непонятно.'
    r = ai(p).lower().strip(' .')
    print(r)
    if r == 'да': return True
    #elif r == 'нет': return False
    return False


def clear_after_stt(text):
    """Получает текст после распознавания из голосового сообщения, пытается его восстановить, исправить ошибки распознавания"""

    # не работает пока что нормально
    return text

    # если нет ключа то сразу отбой
    # if not openai.api_key: return text

    prompt = f'Исправь явные ошибки распознавания голосового сообщения. \
Важна точность, лучше оставить ошибку неисправленной если нет уверенности в том что это ошибка и её надо исправить именно так. \
Если в тексте есть ошибки согласования надо сделать что бы не было. \
Маты и другой неприемлимый для тебя контент переделай так что бы смысл передать другими словами. \
Грубый текст исправь. \
Покажи результат без оформления и своих комментариев. Текст:{prompt}'
    try:
        r = ai(prompt)
    except Exception as e:
        print(e)
        return text
    return r


def check_and_fix_text(text):
    """пытаемся исправить странную особенность пиратского GPT сервера (только pawan?),
    он часто делает ошибку в слове, вставляет 2 вопросика вместо буквы"""

    # для винды нет enchant?
    if 'Windows' in utils.platform():
        return text

    ru = enchant.Dict("ru_RU")

    # убираем из текста всё кроме русских букв, 2 странных символа меняем на 1 что бы упростить регулярку
    text = text.replace('��', '⁂')
    russian_letters = re.compile('[^⁂а-яА-ЯёЁ\s]')
    text2 = russian_letters.sub(' ', text)
    
    words = text2.split()
    for word in words:
        if '⁂' in word:
            suggestions = ru.suggest(word)
            if len(suggestions) > 0:
                text = text.replace(word, suggestions[0])

    # если не удалось подобрать слово из словаря то просто убираем этот символ, пусть лучше будет оопечатка чем мусор
    return text.replace('⁂', '')


def zip_text(text: str) -> str:
    """
    Функция для удаления из текста русских и английских гласных букв типа "а", "о", "e" и "a".
    Так же удаляются идущие подряд одинаковые символы
    """
    vowels = [  'о', 'О',        # русские
                'o', 'O']        # английские. не стоит наверное удалять слишком много

    # заменяем гласные буквы на пустую строку, используя метод translate и функцию maketrans
    text = text.translate(str.maketrans('', '', ''.join(vowels)))

    # убираем повторяющиеся символы
    # используем генератор списков для создания нового текста без повторов
    # сравниваем каждый символ с предыдущим и добавляем его, если они разные 
    new_text = "".join([text[i] for i in range(len(text)) if i == 0 or text[i] != text[i-1]])
    
    return new_text


def query_file(query: str, file_name: str, file_size: int, file_text: str) -> str:
    """
    Query a file using the chatGPT model and return the response.

    Args:
        query (str): The query to ask the chatGPT model.
        file_name (str): The name of the file.
        file_size (int): The size of the file in bytes.
        file_text (str): The content of the file.

    Returns:
        str: The response from the chatGPT model.
    """

    msg = f"""Ответь на запрос юзера по содержанию файла
Запрос: {query}
Имя файла: {file_name}
Размер файла: {file_size}
Текст из файла:


{file_text}
"""
    msg_size = len(msg)
    if msg_size > 99000:
        msg = msg[:99000]
        msg_size = 99000

    result = ''

    if msg_size < 15000:
        try:
            result = ai(msg, model_to_use = 'gpt-3.5-turbo-16k')
        except Exception as error:
            print(error)
            my_log.log2(f'gpt_basic:query_file: {error}')

    if not result and msg_size < 30000:
        try:
            result = ai(msg, model_to_use = 'claude-2-100k')
        except Exception as error:
            print(error)
            my_log.log2(f'gpt_basic:query_file: {error}')

    if not result and msg_size <= 99000:
        try:
            result = ai(msg, model_to_use = 'claude-instant-100k')
        except Exception as error:
            print(error)
            my_log.log2(f'gpt_basic:query_file: {error}')

    if not result:
        try:
            result = ai(msg[:15000], model_to_use = 'gpt-3.5-turbo-16k')
        except Exception as error:
            print(error)
            my_log.log2(f'gpt_basic:query_file: {error}')

    return result


def ai_test() -> str:
    """
    Generates a response using the testing OpenAI ChatCompletion API.

    Returns:
        str: The generated response.
    """
    openai.api_key = cfg.key_test
    openai.api_base = cfg.openai_api_base_test

    # for i in openai.Model.list()['data']:
    #     print(i['id'])
    # return

    #   text = open('1.txt', 'r').read()[:20000]
    text = 'Привет как дела'

    messages = [{"role": "system", "content": "Ты искусственный интеллект отвечающий на запросы юзера."},
                {"role": "user", "content": text}]

    current_model = cfg.model_test

    # тут можно добавить степень творчества(бреда) от 0 до 1 дефолт - temperature=0.5
    сompletion = openai.ChatCompletion.create(
        model = current_model,
        messages=messages,
        max_tokens=2000,
        temperature=0.5,
        timeout=180,
        stream=False
    )
    return сompletion["choices"][0]["message"]["content"]


def stt_after_repair(text: str) -> str:
    query = f"""Исправь текст, это аудиозапись, в ней могут быть ошибки распознавания речи.
Надо переписать так что бы было понятно что хотел сказать человек и оформить удобно для чтения, разделить на абзацы,
добавить комментарии в неоднозначные места.


{text}
"""
    result = ai(query, model_to_use = 'gpt-3.5-turbo-16k')
    return result


def stt(audio_file: str) -> str:
    """
    Transcribes an audio file to text using OpenAI API.

    Args:
        audio_file (str): The path to the audio file.

    Returns:
        str: The transcribed text.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """

    #список серверов на которых доступен whisper
    servers = [x for x in cfg.openai_servers if x[2]]

    assert len(servers) > 0, 'No openai whisper servers configured'

    audio_file_new = Path(utils.convert_to_mp3(audio_file))
    audio_file_bytes = open(audio_file_new, "rb")

    # копируем и перемешиваем список серверов
    shuffled_servers = servers[:]
    random.shuffle(shuffled_servers)

    for server in shuffled_servers:
        openai.api_base = server[0]
        openai.api_key = server[1]
        try:
            translation = openai.Audio.transcribe("whisper-1", audio_file_bytes)
            if translation:
                break
        except Exception as error:
            print(error)
            my_log.log2(f'gpt_basic:stt: {error}\n\nServer: {server[0]}')

    try:
        audio_file_new.unlink()
    except PermissionError:
        print(f'gpt_basic:stt: PermissionError \n\nDelete file: {audio_file_new}')
        my_log.log2(f'gpt_basic:stt: PermissionError \n\nDelete file: {audio_file_new}')

    return json.loads(json.dumps(translation, ensure_ascii=False))['text']


def image_gen(prompt: str, amount: int = 10, size: str ='1024x1024'):
    """
    Generates a specified number of images based on a given prompt.

    Parameters:
        - prompt (str): The text prompt used to generate the images.
        - amount (int, optional): The number of images to generate. Defaults to 10.
        - size (str, optional): The size of the generated images. Must be one of '1024x1024', '512x512', or '256x256'. Defaults to '1024x1024'.

    Returns:
        - list: A list of URLs pointing to the generated images.
    """

    #список серверов на которых доступен whisper
    servers = [x for x in cfg.openai_servers if x[3]]

    assert len(servers) > 0, 'No openai servers with image_gen=True configured'

    prompt_tr = ''
    try:
        prompt_tr = ai_instruct(f'Translate into english if it is not english, else leave it as it is: {prompt}')
    except Exception as image_prompt_translate:
        my_log.log2(f'gpt_basic:image_gen:translate_prompt: {str(image_prompt_translate)}\n\n{prompt}')
    prompt_tr = prompt_tr.strip()
    if not prompt_tr:
        try:
            prompt_tr = my_trans.translate(prompt, 'en')
        except Exception as google_translate_error:
            my_log.log2(f'gpt_basic:image_gen:translate_prompt:google_translate: {str(google_translate_error)}\n\n{prompt}')
        if not prompt_tr:
            prompt_tr = prompt

    assert amount <= 10, 'Too many images to gen'
    assert size in ('1024x1024','512x512','256x256'), 'Wrong image size'

    results = []
    for server in servers:
        openai.api_base = server[0]
        openai.api_key = server[1]
        try:
            response = openai.Image.create(
                prompt = prompt_tr,
                n = amount,
                size=size,
            )
            if response:
                results += [x['url'] for x in response["data"]]
        except AttributeError:
            pass
        except Exception as error:
            print(error)
            my_log.log2(f'gpt_basic:image_gen: {error}\n\nServer: {server[0]}')
    return results


def get_list_of_models():
    """
    Retrieves a list of models from the OpenAI servers.

    Returns:
        list: A list of model IDs.
    """
    result = []
    for server in cfg.openai_servers:
        openai.api_base = server[0]
        openai.api_key = server[1]
        try:
            model_lst = openai.Model.list()
            for i in model_lst['data']:
                result += [i['id'],]
        except Exception as error:
            print(error)
            my_log.log2(f'gpt_basic:get_list_of_models: {error}\n\nServer: {server[0]}')
    return sorted(list(set(result)))


def moderation(text: str) -> str:
    """
    This function performs moderation on the given text.

    Parameters:
        text (str): The text to be moderated.

    Returns:
        str: The result of the moderation process, which may contain categories of flagged content.
    """
    for server in cfg.openai_servers:
        openai.api_base = server[0]
        openai.api_key = server[1]

        try:
            response = openai.Moderation.create(input=text)
            if response:
                result = response['results'][0]['flagged']
                break
        except Exception as error:
            print(error)
            my_log.log2(f'gpt_basic.moderation: {error}\n\nServer: {openai.api_base}')

    categories = response['results'][0]['categories']
    result = ''

    if categories['sexual']:
        result += 'сексуальное содержание, '
    if categories['hate']:
        result += 'ненависть, '
    if categories['harassment']:
        result += 'домогательства, '
    if categories['self-harm']:
        result += 'самоповреждение, '
    if categories['sexual/minors']:
        result += 'сексуальный контент с несовершеннолетними, '
    if categories['hate/threatening']:
        result += 'ненависть/угрозы, '
    if categories['violence/graphic']:
        result += 'насилие/эксплицитный контент, '
    if categories['self-harm/intent']:
        result += 'намерение причинить себе вред, '
    if categories['self-harm/instructions']:
        result += 'инструкции по причинению себе вреда, '
    if categories['harassment/threatening']:
        result += 'домогательства/угрозы, '
    if categories['self-harm/intent']:
        result += 'причинение себе вреда, '

    if result.endswith(', '):
        result = result[:-2]

    return result


if __name__ == '__main__':

    # print(moderation('я тебя убью'))

    # for x in range(5, 15):
    #    print(ai(f'1+{x}='))

    # print(image_gen('большой бадабум'))
    # print(get_list_of_models())

    print(ai('1+1'))