from django.shortcuts import render, redirect
from django.http import JsonResponse
#import openai
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.contrib import auth
from django.contrib.auth.models import User
from .models import Chat
import time, requests
from django.contrib.auth import authenticate

from django.utils import timezone
import shutil
from pydub import AudioSegment
from bs4 import BeautifulSoup
#from googletrans import Translator
import datetime
import random
import re
from django_chatbot import settings
from django_chatbot.settings import MEDIA_URL
import secrets
import string
import pytz
from django.contrib import messages
from datetime import datetime, timedelta
from constance import config
#from claudia import search_from_claudia
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from chatbot.forms import *
from chatbot.speech2text import WhisperModel
import pytz
import os

whisper_model = WhisperModel()
anthropic = Anthropic(api_key='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')
navbar = [
          # {'title': 'Кирүү', 'url': 'login'},
          {'title': 'Тексттен текстке', 'url': 'chatbot'},
        {'title': 'Үндөн үнгө', 'url': 'speech'},
    {'title': 'Тексттен үнгө', 'url': 'text_to_speech'},{'title': 'Үндөн текстке', 'url': 'speech_to_text'},
          ]
def home(request):
    title = 'Башкы бет'

    context = {
        'title': title,
        'navbar': navbar,
    }
    return render(request, 'chatbot/index.html', context=context)


@csrf_protect
@csrf_exempt
def text_to_speech(request):
    context = {}
    if request.method == 'POST':
        text = request.POST['text']
        print(text)
        choose = request.POST['choose']
        form = TextForm(request.POST)
        if form.is_valid():
            request.session['text'] = text
            api_url = 'http://127.0.0.1:6000/api/receive_data'  # Replace with your Flask API's URL
            if choose == 'man':
                data_to_send = {"gender": "man", "text": text}  # Replace with your data

                response = requests.post(api_url, json=data_to_send)
                if response.status_code == 200:
                    received_data = response.json()
                    # print(received_data)
                    # Process the received data as needed
                    old_path = '/mnt/ks/Works/bot_text2speech/' + str(received_data['audio_url']) + '.mp3'
                    file_name = str(received_data['audio_url'])[7:]
                    new_path = '/mnt/ks/Works/ChatbotAll/ChatbotHttps/django_chatbot/media/' + str(
                        file_name) + '.mp3'
                    shutil.move(old_path, new_path)
                    request.session['audio_url'] = file_name + '.mp3'
                    #Audios.objects.create(text=request.session['text'], audio_file=request.session['audio_url'])
                    request.session['audio_url'] = MEDIA_URL + request.session['audio_url']
                    response_data = {'audio_url': request.session['audio_url'], 'audio_text': request.session['text'],
                                     'is_not_valid':
                                         False, 'captcha_error': False, 'other_error': False, 'man': True}
                    return JsonResponse(response_data)
                else:
                    # Handle API error gracefully
                    return JsonResponse({"error": "Failed to send/receive data"}, status=500)

            else:

                data_to_send = {"gender": "woman", "text": text}

                response = requests.post(api_url, json=data_to_send)
                if response.status_code == 200:
                    received_data = response.json()
                    # print(received_data)
                    # Process the received data as needed
                    old_path = '/mnt/ks/Works/bot_text2speech/' + str(received_data['audio_url']) + '.mp3'
                    file_name = str(received_data['audio_url'])[6:]
                    new_path = '/mnt/ks/Works/ChatbotAll/ChatbotHttps/django_chatbot/media/' + str(
                        file_name) + '.mp3'
                    shutil.move(old_path, new_path)
                    request.session['audio_url'] = file_name + '.mp3'
                    #Audios.objects.create(text=request.session['text'], audio_file=request.session['audio_url'])
                    request.session['audio_url'] = MEDIA_URL + request.session['audio_url']
                    response_data = {'audio_url': request.session['audio_url'], 'audio_text': request.session['text'],
                                     'is_not_valid':
                                         False, 'captcha_error': False, 'other_error': False, 'man': False}
                    return JsonResponse(response_data)
                else:
                    # Handle API error gracefully
                    return JsonResponse({"error": "Failed to send/receive data"}, status=500)
        else:
            # print(form.errors)
            if str(form.errors) in captcha_error:
                # print('captcha error')
                response_data = {'captcha_error': True, 'is_not_valid': False, 'other_error': False}
                return JsonResponse(response_data)
            elif str(form.errors) == text_error:
                response_data = {'is_not_valid': True, 'other_error': False}
                return JsonResponse(response_data)
            else:
                response_data = {'other_error': True}
                return JsonResponse(response_data)

    else:
        form = TextForm(request.POST)
    context = {
        'title': 'Тексттен үнгө',
        'navbar': navbar,
        'form': form,
    }
    return render(request, "chatbot/audio.html", context=context)


@csrf_protect
@csrf_exempt
def speech_to_text(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        upload_dir = ''
        N = 4

        res1 = ''.join(secrets.choice(string.ascii_lowercase + string.digits)
                       for i in range(N))
        res2 = ''.join(secrets.choice(string.ascii_lowercase + string.digits)
                       for i in range(N))
        audio_file_name = str(audio_file)[:-4] + '_' + str(res1) + '_' + str(res2) + '.wav'
        file_path = os.path.join(settings.MEDIA_ROOT, upload_dir, audio_file_name)

        # with open(file_path, 'wb') as destination:
        #    for chunk in audio_file.chunks():
        #        destination.write(chunk)

        request.session['audio_url_record'] = upload_dir + audio_file_name

        sound = AudioSegment.from_file(audio_file)
        sound = sound.set_frame_rate(16000)
        sound.export(settings.MEDIA_ROOT + str(request.session['audio_url_record'][:-4]) + '_new.wav', format="wav")
        request.session['audio_url_record'] = str(request.session['audio_url_record'][:-4]) + '_new.wav'

        # st = time.time()
        request.session['text_record'] = whisper_model.generate_text_from_audio(
            settings.MEDIA_ROOT + request.session['audio_url_record'])
        # et = time.time()
        # elapsed_time = et - st
        # print('Execution time:', elapsed_time, 'seconds')
        print(request.session['text_record'])
        Audios.objects.create(audio_file=request.session['audio_url_record'], text=request.session['text_record'])
        request.session['audio_url_record'] = MEDIA_URL + request.session['audio_url_record']
        request.session['audio_save'] = True
        context = {
            'title': 'Үндөн текстке',
            'navbar': navbar,
            'is_not_valid': False,
            'record': request.session['audio_url_record'],
            'record_text': request.session['text_record']
        }
        return JsonResponse(
            {'success': True, 'audio_url': request.session['audio_url_record'], 'text': request.session['text_record']})
    else:
        if 'audio_save' in request.session and request.session['audio_save']:
            context = {
                'title': 'Үндөн текстке',
                'navbar': navbar,
                'record': request.session['audio_url_record'],
                'record_text': request.session['text_record']
            }
            request.session['audio_save'] = False
            return render(request, "chatbot/audio2.html", context=context)

        context = {
            'title': 'Үндөн текстке',
            'navbar': navbar,
            'record': False,
            'record_text': False
        }
        return render(request, "chatbot/audio2.html", context=context)
def search_from_claudia(text):

    text = '\n\nHuman: ' + 'кыргызча жооп бер ' + '\n\nAssistant: Макул кыргызча жооп берем\n\nHuman:' + text
    stream = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=200,

        prompt=f"{HUMAN_PROMPT}{text}{AI_PROMPT}",
        stream=True
    )
    res = ''
    for complation in stream:
        res += complation.completion
        print(complation.completion, end='', flush=True)
    print(res)
    return res
def translate_text(text, target_language='ky'):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text
def search_on_mistal(text):
    #print(1221)
    api_url = 'http://80.72.180.130:3050/api/receive_data'  # Replace with your Flask API's URL
    #text = 'Ты кто такой?'

    if not text[-1] == '?':
        text = text + '?'
    l = len(text)
    data_to_send = {
        "in_text": text}  # Replace with your data
    response = requests.post(api_url, json=data_to_send)

    if response.status_code == 200:
        received_data = response.json()
        out = str(received_data['out_text'])
        new_out = out.strip()
        #print(out)
        #
        new_out = new_out.replace('<s>', '')
        new_out = new_out.replace('[/INST] ', '')
        new_out = new_out.replace('[INST] ', '')
        new_out = new_out[l:]
        new_out = re.split('(?<=[\.\?\!])\s*', new_out)
        #print(new_out)
        ans = ''
        for i in new_out:
            if i.endswith('?'):
                ans = ans + str(i)
            else:
                ans = ans + str(i)



        return ans
        # print(request.session['text_record'])
    else:
        return 'ката чыкты'

def search_on_internet(text):
    api_url = "https://google.serper.dev/search"
    search_payload = {
        "q": text,
        "gl": "kg",
        "hl": "ky",
    }
    api_headers = {
        'X-API-KEY': '22a5f83e46688f97b9b60036fa3abf0d8222641b',
        'Content-Type': 'application/json'
    }

    response = requests.post(api_url, json=search_payload, headers=api_headers)
    json_data = response.json()



# Извлечение информации из объекта JSON
    search_parameters = json_data.get("searchParameters", {})
    organic_results = json_data.get("organic", [])

    # Вывод сниппетов только для результатов с "position": 1
    for result in organic_results:
        for i in range(20):
            position = result.get("position", 0)
            if position ==i:
                snippet = result.get("snippet", "")
                #print(f'snippets length: {len(snippet)}')
                link = result.get("link", "")
                #print("snip",snippet)
                #print("snipet13", snippet[:13])
                # Отправка GET-запроса по полученному линку
                response = requests.get(link)
                page_content = response.text
               # print(link)
# Джала́л-Аба́д[3] (кирг. Жалал-Абад, Жалалабат, сокр. ДЖл) — третий по величине город в Киргизии, административный центр Джалал-Абадской области. Население — 129 400 человек[1]. Расположен на юге страны в Ферганской долине. Недалеко от города протекает река Кугарт, правый приток реки Кара-Дарья.


                # Использование BeautifulSoup для анализа HTML страницы
                soup = BeautifulSoup(page_content, 'html.parser')
                paragraphs = [re.sub(r'<.*?>', '', paragraph.get_text()) for paragraph in soup.find_all('p')]
                for paragraph in paragraphs:
                    #print('old')
                    #print(paragraph)
                    paragraph = re.sub(r'\[\d+\]', '', paragraph)
                    #paragraph = paragraph.replace('^[0-9]+$','')
                    #print('new')
                    #print(paragraph)
                    #j = 20
                    for j in range(len(snippet), len(snippet)-20, -1):

                        if snippet[:j] in paragraph:
                            #print("res", paragraph)
                            return paragraph
                        else:
                            #print('!')
                            #print(paragraph)
                            continue

                break
            else:
                i = i + 1
    return "Мындай маалымат жок"




    # api_key = 'AIzaSyAPEnSDbWyRbnajVbpithCsn4jkXNG6LF0'
    # cx = '40239d5a01b394fe0'  # Replace with your actual Custom Search Engine ID
    #
    # url = f'https://www.googleapis.com/customsearch/v1?q={text}&key={api_key}&cx={cx}&hl=ky&num=1'
    #
    # try:
    #
    #     response = requests.get(url)
    #     response.raise_for_status()
    #     data = response.json()
    #     print(data)
    #     if 'items' in data and data['items']:
    #         first_result = data['items'][0]
    #         return [first_result]
    #     else:
    #         return "ката чыкты"
    #
    # except requests.exceptions.RequestException as e:
    #     return f"Error during internet search: {str(e)}"
def get_data_from_tts(request, gender, text, audio_url):
    api_url = 'http://127.0.0.1:6000/api/receive_data'  # Replace with your Flask API's URL
    data_to_send = {"gender": gender, "text": text}  # Replace with your data

    response = requests.post(api_url, json=data_to_send)
    if response.status_code == 200:
        received_data = response.json()
        # print(received_data)
        # Process the received data as needed
        old_path = '/mnt/ks/Works/bot_text2speech/' + str(received_data['audio_url']) + '.mp3'
        # print(received_data['audio_url'])
        if gender == 'woman':
            file_name = str(received_data['audio_url'])[6:]
        else:
            file_name = str(received_data['audio_url'])[7:]
        new_path = '/mnt/ks/Works/Chatbot/django_chatbot/media/' + str(
            file_name) + '.mp3'
        shutil.move(old_path, new_path)
        request.session[audio_url] = file_name + '.mp3'
    else:
        print('error')
        return JsonResponse({"error": "Failed to send/receive data"}, status=500)
    return request.session[audio_url]
def get_data_from_whisper(requests, url):
    api_url = 'http://127.0.0.1:7000/api/receive_data'  # Replace with your Flask API's URL
    data_to_send = {
        "audio_url": url}  # Replace with your data
    response = requests.post(api_url, json=data_to_send)
    if response.status_code == 200:
        received_data = response.json()
        text = str(received_data['text'])
        # print(request.session['text_record'])
    else:
        print('error in whisper')
        return JsonResponse({"error": "Failed to send/receive data"}, status=500)
    return text


@csrf_protect
@csrf_exempt
def speech(request):
    context = {}
    if request.method == 'POST':

        audio_file = request.FILES['audio2']
        #upload_dir = 'recorded/'
        N = 4
        res1 = ''.join(secrets.choice(string.ascii_lowercase + string.digits)
                       for i in range(N))
        res2 = ''.join(secrets.choice(string.ascii_lowercase + string.digits)
                       for i in range(N))
        audio_file_name = str(audio_file)[:-4] + '_' + str(res1) + '_' + str(res2) + '.wav'
        #file_path = os.path.join(settings.MEDIA_ROOT, upload_dir, audio_file_name)
        # with open(file_path, 'wb') as destination:
        #    for chunk in audio_file.chunks():
        #        destination.write(chunk)
        print(audio_file_name)
        request.session['audio_url_record'] = audio_file_name
        sound = AudioSegment.from_file(audio_file)
        sound = sound.set_frame_rate(16000)
        sound.export(settings.MEDIA_ROOT + str(request.session['audio_url_record'][:-4]) + '_new.wav', format="wav")
        request.session['audio_url_record'] = str(request.session['audio_url_record'][:-4]) + '_new.wav'
        # print(request.session['audio_url_record'])
        # print(settings.MEDIA_ROOT + request.session['audio_url_record'])
        # st = time.time()

        # request.session['text_record'] = whisper_model.generate_text_from_audio(settings.MEDIA_ROOT+request.session['audio_url_record'])
        request.session['text_record'] = get_data_from_whisper(requests,
            settings.MEDIA_ROOT + request.session['audio_url_record'])

        # et = time.time()
        # elapsed_time = et - st
        # print('Execution time:', elapsed_time, 'seconds')

        #Audios.objects.create(audio_file=request.session['audio_url_record'], text=request.session['text_record'])

        request.session['audio_url_record'] = MEDIA_URL + request.session['audio_url_record']

        print(f'Input: {request.session["text_record"]}')

        res = search_from_claudia(request.session['text_record'])
        print(f'Answer from Claudia: {res}')
        # res = search_on_mistal(request.session['text_record'])
        # print(f'Answer from Mistral: {res}')
        # res_inter = search_on_internet(request.session['text_record'])
        # print(f'Answer from Internet: {res_inter}')
        #res_inter = translate_text(res_inter)
        #print(f'Translated: {res_inter}')

        if request.POST['choose'] == 'man':

            request.session['audio_url'] = get_data_from_tts(request, 'man', res, 'audio_url')
            # Audios.objects.create(text=request.session['text'], audio_file=request.session['audio_url'])
            request.session['audio_url'] = MEDIA_URL + request.session['audio_url']

            # request.session['audio_url2'] = get_data_from_tts(request, 'man', res, 'audio_url2')
            # request.session['audio_url2'] = MEDIA_URL + request.session['audio_url2']
            # print(request.session['audio_url2'])

            response_data = {'success': True, 'audio_url_output': request.session['audio_url'],
                             #'audio_url_output2': '',
                             'audio_url_input': request.session['audio_url_record']
                             }
            return JsonResponse(response_data)


        else:
            request.session['audio_url'] = get_data_from_tts(request, 'woman', res, 'audio_url')
            # Audios.objects.create(text=request.session['text'], audio_file=request.session['audio_url'])
            request.session['audio_url'] = MEDIA_URL + request.session['audio_url']

            # request.session['audio_url2'] = get_data_from_tts(request, 'woman', res, 'audio_url2')
            # request.session['audio_url2'] = MEDIA_URL + request.session['audio_url2']
            # print(request.session['audio_url2'])

            response_data = {'success': True, 'audio_url_output': request.session['audio_url'],
                             #'audio_url_output2': '',
                             'audio_url_input': request.session['audio_url_record']
                             }
            return JsonResponse(response_data)

        # request.session['audio_save'] = True

    else:

        context = {
            'title': 'Үндөн үнгө',
            'navbar': navbar
        }
        return render(request, "chatbot/audio4.html", context=context)
# Create your views here.
@csrf_protect
@csrf_exempt
def chatbot(request):
    #
    current_time = datetime.now(pytz.timezone('Etc/GMT-6'))
    #print(f'log time : {datetime.fromisoformat(request.session["login_time"])}')
    #one_day_later = datetime.fromisoformat(request.session['login_time']) + timedelta(minutes=60)

    #print(one_day_later)
    # , created_at__lt=one_day_later
    # chats = Chat.objects.filter(user=request.user.id, created_at__gte=datetime.fromisoformat(request.session['login_time']), created_at__lt=one_day_later)
    # if current_time > datetime.fromisoformat(request.session['login_time']) and current_time < one_day_later:
    #
    #     chats = Chat.objects.filter(user=request.user.id,
    #                                 created_at__range=(datetime.fromisoformat(request.session['login_time']),
    #                                                    one_day_later))
    #
    # else:
    #     current_time = datetime.now(pytz.timezone('Etc/GMT-6'))
    #     current_time_str = current_time.isoformat()
    #     request.session['login_time'] = current_time_str
    #     chats = ''
    chats = ''


    if request.method == 'POST':
        request.session['message'] = request.POST['message']
        my_setting_value = config.Model
        print(my_setting_value)
        #print(my_setting_value)
        #response = ask_openai(message)
        #messages.success(request, 'Your message was successfully processed.')
        #print(request.session['message'])
        # if not request.user.is_authenticated:
        #     response = 'Сураныч каттоодон өтүңүз же кириңиз'
        #     current_time = datetime.now(pytz.timezone('Etc/GMT-6'))
        #     current_time = str(current_time)[:-13]
        #     return JsonResponse({'message': request.session['message'], 'response': response, 'time': current_time})
        # time.sleep(1)
        # response = 'hi'
        if my_setting_value == 'Mistral':
            #response = search_on_mistal(request.session['message'])
            #response = 'Hi'
            current_time = datetime.now(pytz.timezone('Etc/GMT-6'))
            current_time = str(current_time)[:-13]
            # chat = Chat(user=request.user, message=request.session['message'], response=response)
            # chat.save()
            return JsonResponse({'message': request.session['message'], 'response': 'response', 'model': 'Mistral', 'time': current_time})
        elif my_setting_value == 'Claudia':
            #com = search_from_claudia(request.POST['message'])
            current_time = datetime.now(pytz.timezone('Etc/GMT-6'))
            current_time = str(current_time)[:-13]
            return JsonResponse({'message': request.session['message'], 'response': 'response', 'model': 'Claudia', 'time': current_time})

        elif my_setting_value == 'Random':
            variable1 = "Mistral"
            variable2 = "Claudia"
            random_choice = random.choice([variable1, variable2])
            if random_choice == 'Claudia':
                current_time = datetime.now(pytz.timezone('Etc/GMT-6'))
                current_time = str(current_time)[:-13]
                return JsonResponse(
                    {'message': request.session['message'], 'response': 'response', 'model': 'Claudia', 'time': current_time})
            else:
                #Mistral
                response = search_on_mistal(request.session['message'])
                #response = 'Hi'
                current_time = datetime.now(pytz.timezone('Etc/GMT-6'))
                current_time = str(current_time)[:-13]
                chat = Chat(user=request.user, message=request.session['message'], response=response)
                chat.save()
                return JsonResponse(
                    {'message': request.session['message'], 'response': response, 'model': 'Mistral',
                     'time': current_time})

        #print(f'Answer from Mistral: {response}')
        # print(f'Answer from Mistral: {res_mistral}')
        # response = search_on_internet(request.session['message'])
        # print(response)
        #response = translate_text(response)


        #return JsonResponse({'message': request.session['message'], 'response': response, 'time': current_time})
    context = {
        'title': 'Тексттен текстке',
        'navbar': navbar,
        'chats': chats
    }
    return render(request, 'chatbot/chatbot.html', context=context)
    #return render(request, 'chatbot/chatbot.html', {'chats': chats})

@csrf_protect
@csrf_exempt
def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            current_time = datetime.now(pytz.timezone('Etc/GMT-6'))

            # Convert datetime to string
            current_time_str = current_time.isoformat()
            #print(f'login time: {current_time_str}')
            # Save the string to the session variable
            request.session['login_time'] = current_time_str

            return redirect('home')
        else:
            error_message = 'Колдонуучунун аты же сыр сөз туура эмес'
            return render(request, 'chatbot/login.html', {'error_message': error_message})
    else:
        return render(request, 'chatbot/login.html')


@csrf_protect
@csrf_exempt
def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:
            if len(password1) >= 8:
                if User.objects.filter(username=username).exists():
                    error_message = 'Колдонуучу мындай аты менен бар'
                    return render(request, 'chatbot/register.html', {'error_message': error_message})
                else:
                    user = User.objects.create_user(username, email, password1)
                    user.save()
                    auth.login(request, user)
                    current_time = datetime.now(pytz.timezone('Etc/GMT-6'))

                    # Convert datetime to string
                    current_time_str = current_time.isoformat()
                    # print(f'login time: {current_time_str}')
                    # Save the string to the session variable
                    request.session['login_time'] = current_time_str
                    return redirect('home')
            else:
                error_message = 'Сыр сөз эң аз 8 символдон туруш керек'
                return render(request, 'chatbot/register.html', {'error_message': error_message})
        else:
            error_message = 'Сыр сөз дал келген жок'
            return render(request, 'chatbot/register.html', {'error_message': error_message})
    return render(request, 'chatbot/register.html')

def logout(request):
    auth.logout(request)
    return redirect('login')
