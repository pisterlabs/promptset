from ttsModule import VoiceModel
from translatorModule import TranslatorModel
from openAIModule import TextNeuralNetwork as gpt
from sttModule import SpeachToText as speachToText, listen
from SettingsManager import SettingsManager, settings_manager
from fuzzywuzzy import fuzz
from notificationModule import show_notification
from datetime import datetime
from writeCommand import listen_write
from multiprocessing import Process
from tkinter import Scrollbar, Text, Frame
import random
import tkinter as tk
import time
import asyncio
import subprocess
import platform
import webbrowser
import inflect


settings_manager.load_settings()

# Asynchronous function for request in chatGPT
async def gpt_req(text_to_gpt: str):
    freeGPT = gpt()
    return await freeGPT.create_prompt(text_to_gpt)


# The function converts numbers to a string
def convert_numbers_in_text(text: str):
    p = inflect.engine()
    words = text.split()
    for i in range(len(words)):
        try:
            num = int(words[i])
            words[i] = p.number_to_words(num)
        except ValueError:
            pass
    return ' '.join(words)


# The function creates a tkinder window to display the string
def paint_answer(answer_text: str):
    root = tk.Tk()
    root.title("Відповідь")

    frame = Frame(root)
    frame.pack(fill='both', expand=True)

    text_widget = Text(frame, wrap='word', width=60, height=15)
    text_widget.insert('1.0', answer_text)

    scroll = Scrollbar(frame, command=text_widget.yview)
    text_widget.config(yscrollcommand=scroll.set)

    text_widget.grid(row=0, column=0, sticky='nsew')
    scroll.grid(row=0, column=1, sticky='ns')

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    root.geometry("500x500")
    root.mainloop()


# Function creates a process to display a string via tkinter
def paint_gpt_answer(text: str):
    window_process = Process(target=paint_answer, args=[text])
    window_process.start()


# function of opening a folder or executable file of a custom command
def execute_custom_command_exe(ff_path: str, type: str):
    try:
        if type == 'file':
            result = subprocess.run([ff_path], check=True, text=True, capture_output=True)
        if type == 'path':
            system_type = platform.system()
            if system_type == "Windows":
                result = subprocess.run(['explorer', ff_path], check=True)
                show_notification("Голосовий помічник","Теку выдкрито")
            elif system_type == "Linux":
                result = subprocess.run(['xdg-open', ff_path], check=True)
                show_notification("Голосовий помічник","Теку выдкрито")

    except subprocess.CalledProcessError as e:
        #print("Error:", e)
        pass


# the function translates all spoken text into a user request without a command and an alias assistant
def make_req_from_string(original_string: str, key_world: str):
    key_world_count = len(key_world.split())
    words = original_string.split()
    remaining_words = words[key_world_count+1:]
    result_string = ' '.join(remaining_words)
    return result_string


# making the textToSpeach model
def make_tss_model(lang: str, tts: str):
    language = tts[lang]['language']
    model_id = tts[lang]['model_id']
    sample_rate = tts[lang]['sample_rate']
    speaker = tts[lang]['speaker']
    return VoiceModel(language, model_id, sample_rate, speaker)


# The function translates text into voice
def speech_the_text(text: str, tts_model: VoiceModel):
    tts_model.play_audio(text)
    return 0


# execute command function
def execute_cmd(cmd: str, key_world: str, voice: str, assistant_alias, assistant_cmd_list, assistant_tts, assistant_stt, assistant_tra, current_settings, tts_model: VoiceModel):
    cur_tra_to_lang = current_settings['ASSISTANT_TRA']
    cur_speach_lang = current_settings['ASSISTANT_TTS']
    cur_speaker_lang = current_settings['ASSISTANT_STT']
    is_quick_answer = current_settings['IS_QUICK_ANSWER']
    speak_the_answer = current_settings['SPEAK_THE_ANSWER']


    if cmd == 'help':
    # Add browser opening to documentation
        webbrowser.open('https://github.com/OlegRad4encko/Diploma')


    # command time - makes notifications with time \ or says so
    elif cmd == 'time':
        current_time = datetime.now()
        p = inflect.engine()
        hour_text = p.number_to_words(current_time.hour)
        minute_text = p.number_to_words(current_time.minute)
        time_text = f"'{hour_text} {p.plural('hour', current_time.hour)} and {minute_text} {p.plural('minute', current_time.minute)} at the moment"

        translate = TranslatorModel('en', cur_speach_lang)
        translated_text = translate.translate_text(time_text)

        if speak_the_answer == "False":
            show_notification("Голосовий помічник",translated_text)
        else:
            speech_the_text(translated_text, tts_model)


    # command open browser - opens the browser
    elif cmd == 'open browser':
        webbrowser.open('https://www.google.com.ua/?hl=uk')


    # The joke command - makes a notification with a joke \ or tells a joke
    elif cmd == 'joke':
        jokes = ['Є люди, які несуть в собі щастя. Коли ці люди поруч, все ніби стає яскравим і барвистим. Але моя дружина їх називає алкашами!',
            'З одного боку, в гості без пляшки не підеш. А з іншого якщо в тебе є пляшка, та на холєру в гості пертись?',
            'Кохана, давай миритися. Вибач мене, я був не правий. Стоп! Не їж! Я приготую щось інше!',
            'Кажуть, що у геніїв в будинку має бути безлад. Дивлюсь на свою дитину і гордість розпирає! Енштейна виховую!.',
            'Христос родився! Маю три вакцини, сертифікат вакцинації, негативний тест, оброблений антисептиком! Колядувати можна?',
            'Пішла, куда послав. Веду себе, як ти назвав. І чому я тебе раніше не слухала?!']

        joke = random.choice(jokes)
        translated_text = ''
        if speak_the_answer == "False":
            if cur_speach_lang != 'ua':
                translate = TranslatorModel('uk', cur_speach_lang)
                translated_text = translate.translate_text(joke)
            else:
                translated_text = joke
            show_notification("Голосовий помічник",translated_text)
        else:
            if cur_speach_lang != 'ua':
                translate = TranslatorModel('uk', cur_speach_lang)
                translated_text = translate.translate_text(joke)
            else:
                translated_text = joke

            speech_the_text(translated_text, tts_model)


    # command write - writes to the active window, everything that the user says
    elif cmd == 'write':
        write_process = Process(target=listen_write, args=[assistant_stt[current_settings['ASSISTANT_STT']]['model']])
        write_process.start()

        answer = "Диктуйте"
        if speak_the_answer == "False":
            if cur_speach_lang != 'ua':
                translate = TranslatorModel('uk', cur_speach_lang)
                translated_text = translate.translate_text(answer)
            else:
                translated_text = answer
            time.sleep(5)
            show_notification("Голосовий помічник",translated_text)
        else:
            if cur_speach_lang != 'ua':
                translate = TranslatorModel('uk', cur_speach_lang)
                translated_text = translate.translate_text(answer)
            else:
                translated_text = answer
            time.sleep(6)
            speech_the_text(translated_text, tts_model)


    # command find - opens a browser window with a user request
    elif cmd == 'find':
        va_request = make_req_from_string(voice, key_world)
        url_request = "https://www.google.com/search?q=" + '+'.join(va_request.split())
        webbrowser.open(url_request)
        show_notification("Голосовий помічник","Відкрито браузер з запитом '"+va_request+"'")


    # the say command - either says what the user said \ or notifies that the voice is not turned on
    elif cmd == 'say':
        if speak_the_answer == "False":
            show_notification("Голосовий помічник","Увімкніть озвучування відповідей голосового помічника")

        else:
            speech_the_text(make_req_from_string(voice, key_world), tts_model)


    # makes a request to the GPT chat
    elif cmd == 'gpt':
        va_request = make_req_from_string(voice, key_world)
        show_notification("Голосовий помічник","Очікуйте, звертаюсь до ChatGPT, це може зайняти декілька хвилин")
        result = asyncio.run(gpt_req(va_request))
        result_without_numbers = convert_numbers_in_text(result)

        if len(result_without_numbers) < 500:
            if speak_the_answer == "False":
                if len(result_without_numbers) < 100:
                    show_notification("Голосовий помічник", result)
                else:
                    paint_gpt_answer(result)
            else:
                paint_gpt_answer(result)
                translate = TranslatorModel('en', cur_speach_lang)
                translated_text = translate.translate_text(result_without_numbers)
                speech_the_text(translated_text, tts_model)

        else:
            paint_gpt_answer(result)


    # translate command
    elif cmd == 'translate':
        va_request = make_req_from_string(voice, key_world)
        translate = TranslatorModel(cur_speach_lang, cur_tra_to_lang)
        translated_text = translate.translate_text(va_request)
        result = va_request +f'\u000a\u000a'+translated_text
        paint_gpt_answer(result)


    # custom command
    else:
        if assistant_cmd_list[cmd]['commandType'] == 'explorer':
            execute_custom_command_exe(assistant_cmd_list[cmd]['customCommand'], 'path')
        elif assistant_cmd_list[cmd]['commandType'] == 'execute':
            execute_custom_command_exe(assistant_cmd_list[cmd]['customCommand'], 'file')
        elif assistant_cmd_list[cmd]['commandType'] == 'openWebPage':
            webbrowser.open(f'{assistant_cmd_list[cmd]["customCommand"]}')
        else:
            show_notification("Голосовий помічник", "Я Вас не зрозумів")



## recognizing function v1
# def recognize_cmd(cmd: str, assistant_cmd_list):
#     rc = {'cmd': '', 'percent': 0, 'key_world': ''}
#     max_per = 0
#     key_world = ''
#     for word_list_key in assistant_cmd_list:
#         word_list = assistant_cmd_list[word_list_key]["word_list"]
#         for x in word_list:
#             vrt = fuzz.ratio(cmd, x)
#             if max_per < vrt:
#                 max_per = vrt
#                 key_world = x
#             if vrt > rc['percent']:
#                 rc['cmd'] = word_list_key
#                 rc['percent'] = vrt
#                 rc['key_world'] = key_world
#     return rc



## recognizing function v2
def recognize_cmd(cmd: str, assistant_cmd_list):
    rc = {'cmd': '', 'percent': 0, 'key_world': ''}

    for word_list_key in assistant_cmd_list:
        word_list = assistant_cmd_list[word_list_key]["word_list"]
        max_per, key_world = max((fuzz.partial_ratio(cmd, x), x) for x in word_list)
        if max_per > rc['percent']:
            rc['cmd'] = word_list_key
            rc['percent'] = max_per
            rc['key_world'] = key_world

    return rc


# voice recognition callback function
def stt_respond(voice: str, assistant_alias, assistant_cmd_list, assistant_tts, assistant_stt, assistant_tra, current_settings, tts_model):
    if voice.startswith(assistant_alias):
        cmd = recognize_cmd(filter_cmd(voice, assistant_alias), assistant_cmd_list)
        if cmd['cmd'] not in assistant_cmd_list.keys():
            show_notification("Голосовий помічник", "Я Вас не зрозумів")
        else:
            execute_cmd(cmd['cmd'], cmd['key_world'], voice, assistant_alias, assistant_cmd_list, assistant_tts, assistant_stt, assistant_tra, current_settings, tts_model)


# filtering the CMD, removing еру assistant alias from raw_voice
def filter_cmd(raw_voice: str, assistant_alias):
    cmd = raw_voice

    for x in assistant_alias:
        cmd = cmd.replace(x, "").strip()

    return cmd


# update settings for voice assistant
def update_settings():
    settings_manager.load_settings()
    current_settings = settings_manager.get_setting('CURRENT_SETTINGS', {})
    assistant_cmd_list = settings_manager.get_setting('ASSISTANT_CMD_LIST', {})
    assistant_alias = settings_manager.get_setting('ASSISTANT_ALIAS', {})
    assistant_stt = settings_manager.get_setting('ASSISTANT_STT', {})
    assistant_tts = settings_manager.get_setting('ASSISTANT_TTS', {})
    assistant_tra = settings_manager.get_setting('ASSISTANT_TRA', {})
    current_settings = settings_manager.get_setting('CURRENT_SETTINGS', {})
    return {
        "current_settings": current_settings,
        "assistant_cmd_list": assistant_cmd_list,
        "assistant_alias": assistant_alias,
        "assistant_stt": assistant_stt,
        "assistant_tts": assistant_tts,
        "assistant_tra": assistant_tra,
        "current_settings": current_settings
    }


# Run command processor
def run_command_processor():
    settings = update_settings()
    current_settings = settings['current_settings']
    assistant_cmd_list = settings['assistant_cmd_list']
    assistant_alias_list = settings['assistant_alias']
    assistant_stt = settings['assistant_stt']
    assistant_tts = settings['assistant_tts']
    assistant_tra = settings['assistant_tra']
    current_settings = settings['current_settings']

    cur_speach_lang = current_settings['ASSISTANT_TTS']

    tts_model = make_tss_model(cur_speach_lang, assistant_tts)
    assistant_alias = tuple(assistant_alias_list)

    speachVA = speachToText(assistant_stt[current_settings['ASSISTANT_STT']]['model'])
    listen(stt_respond, speachVA.get_model(), assistant_alias, assistant_cmd_list, assistant_tts, assistant_stt, assistant_tra, current_settings, tts_model)
