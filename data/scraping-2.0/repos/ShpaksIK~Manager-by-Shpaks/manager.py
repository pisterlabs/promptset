"""

Код главного файла программы 'Manager by Shpaks'.
Версия: 1.5.1
Автор: Shpaks
Copyright Manager by Shpaks | 2023

"""
# Импорты
import colorama
import socket
import datetime
import json
import os
import platform
import random
import requests
import sys
import time
import webbrowser
import phonenumbers
from threading import Thread

import asyncio
import folium
import gtts
import importlib
import openai
import pygame
import pyttsx3
import subprocess
import translate

from colorama import Back, Fore
from pyfiglet import Figlet


# Инициализация библиотек
colorama.init()
pygame.mixer.init()
os.system("cls")
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)
device_name = socket.gethostname()

# Основная функция программы
def programm(nick):
    # Инициализания данных программы
    version = "1.5.1"
    running = True
    privileg = ["VIP", "VIP+", "Легенда", "Модератор", "Создатель"]
    commands = ["/help", "/relist", "dellist", "/info", "/list", "/balance", "/bal", "/balance_user", "/spam", "/pay", "/retitle", "/update", "/randpass", "/yt", "/sear", "/bor", "/gdz", "/trr", "/tre", "/kazino", "/or", "/try", "/ac", "/ban", "/unban", "/banlist", "/404", "/deluser", "/color", "/reload", "/exit", "/clear", "/donate", "/promo", "/renick", "/changepassword", "/cp", "/logout", "/ya", "/google", "/give", "/give_balance", "/give_rep", "/give_perm", "/ip", "/addpromo", "/delpromo", "/rep", "/unrep", "/user", "/users", "/friends", "/friend_add", "/friend_del", "/friend_info", "/settings", "/settings_ai_token", "/settings_color", "/settings_bgcolor", "/gpt", "/timer", "/time", "/voice", "/wifi", "/course", "/block_site", "/scripts", "/add_script", "/del_script", "/load_script", "/create_script", "/gpt_site", "/phone"]
    reason_404 = None
    timer_active = False
    messages_cooldown_bool = False
    
    # Получение данных пользователя
    with open('data.json', 'r+', encoding='utf-8') as json_file:
        file_data = json.load(json_file)
        # id пользователя
        temp_id = 0
        for i in file_data["id"]:
            try:
                if file_data["id"][temp_id][nick]["name"] == nick:
                    id = file_data["id"][temp_id][nick]["id"]
            except KeyError:
                pass
            temp_id += 1

    # Выход из сессии, если заблокирован
    with open('data.json', 'r', encoding='utf-8') as jf:
        fd = json.load(jf)
    if fd['user'][id-1]['ban'] is True or fd['user'][id-1]['404'] is True:
        with open('settings.json', 'r', encoding='utf-8') as jf:
            settings = json.load(jf)
        settings['session']['name'] = ""
        settings['session']['password'] = ""
        with open('settings.json', 'w') as jf:
            json.dump(settings, jf, indent=4)

    # Изменения значений
    def comm():
        with open('data.json', 'r', encoding='utf-8') as jf:
            fd = json.load(jf)
        fd["user"][id-1]["comm"] += 1
        with open('data.json', 'w') as jf:
            json.dump(fd, jf, indent=4)

    def renick(new_nick, old_nick=nick):
        with open('data.json', 'r', encoding='utf-8') as jf:
            fd = json.load(jf)
        old_nick = fd["user"][id-1]["name"]
        renick_check = False
        temp_check_renick = 0
        for i in fd["user"]:
            if fd["user"][temp_check_renick]["name"] == new_nick:
                print("[!] Данный никнейм уже используется")
                renick_check = True
                break
            temp_check_renick += 1
        if renick_check is False:
            fd["user"][id-1]["name"] = new_nick
            fd["id"][id-1][new_nick] = fd["id"][id-1].pop(old_nick)
            fd["id"][id-1][new_nick]["name"] = new_nick
            with open('data.json', 'w') as jf:
                json.dump(fd, jf, indent=4)
            print(f"[/] Вы успешно заменили никнейм {old_nick} на {new_nick}")
    
    def changepassword(new_password):
        with open('data.json', 'r', encoding='utf-8') as jf:
            fd = json.load(jf)
        fd["user"][id-1]["password"] = new_password
        with open('data.json', 'w') as jf:
            json.dump(fd, jf, indent=4)
        with open('settings.json', 'r', encoding='utf-8') as jf:
            settings = json.load(jf)
        settings["session"]["name"] = ""
        settings["session"]["password"] = ""
        with open('settings.json', 'w') as jf:
            json.dump(settings, jf, indent=4)
        print("[/] Ваш пароль успешно изменён")

    def addlist(mylist):
        with open('data.json', 'r', encoding='utf-8') as jf:
            fd = json.load(jf)
        fd["user"][id-1]["list"] = mylist
        with open('data.json', 'w') as jf:
            json.dump(fd, jf, indent=4)

    def banned(nick=nick, id=id, reason_404=reason_404):
        os.system("cls")
        with open('data.json', 'r', encoding='utf-8') as jf:
            fd = json.load(jf)
        if fd['user'][id-1]['404'] is False:
            reason = fd['user'][id-1]['reason']
            who_ban = fd['user'][id-1]['who_ban']
            print("======================= БАН =======================")
            print(f"Вы, {nick} ({id}), получили блокировку аккаунта:")
            print(f"Причина: {reason}")
            print(f"Заблокировал: {who_ban}")
            print(f"Время: {datetime.datetime.now().strftime('%D %H:%M:%S')}")
            print("Для восстановления обратитесь в поддержку.")
            print("===================================================")
        else:
            print("======================= БАН 404 =======================")
            print(f"Вы, {nick} ({id}), получили блокировку и отчистку аккаунта.")
            print("Ваша статистика, кроме id, никнейма и пароля была удалена.")
            print(f"Причина: 404")
            print(f"Заблокировал: Создатель")
            print(f"Время: {datetime.datetime.now().strftime('%D %H:%M:%S')}")
            print("Для восстановления обратитесь в поддержку.")
            print("=======================================================")
        input()

    def retitle(new_title):
        with open('data.json', 'r', encoding='utf-8') as jf:
            fd = json.load(jf)
        fd["user"][id-1]["title"] = new_title
        with open('data.json', 'w') as jf:
            json.dump(fd, jf, indent=4)

    while running:
        with open('data.json', 'r', encoding='utf-8') as jf:
            fd = json.load(jf)
        if fd["user"][id-1]["ban"] is True or fd["user"][id-1]["404"] is True:
            banned()
            break
        try:
            try:
                command = input()
                command_list = str(command)
                if command.replace(" ", "") == "":
                    pass
                else:
                    if command_list[0] == "/":
                        if command in commands:
                            comm() 
                        if command == "/help":
                            print("============ Список команд и их описание ============")
                            print("ОБЩИЕ, ОСНОВНЫЕ КОММАНДЫ:")
                            print(" /help - данный список команд")
                            print(" /v - версия программы")
                            print(" /donate - список привилегий")
                            print(" /ex - прекратить использвание комманды")
                            print(" /relist - добавляет записи")
                            print(" /dellist - удаляет запись")
                            print(" /list - просмотр записей")
                            print(" /retitle - изменить свой титул")
                            print(" /balance - узнать свой баланс")
                            print(" /balance_user - узнать баланс другого пользователя")
                            print(" /pay - заплатить другому пользователю указанное количество денег")
                            print(" /rep - выдаёт репутацию указанному пользователю")
                            print(" /unrep - отнимает репутацию указанного пользователя")
                            print(" /info - показывают вашу статистику")
                            print(" /update - просмотр последих обновлений")
                            print(" /promo - активировать промокод")
                            print(" /user - узнать информацию о пользователе")
                            print(" /users - список зарегистрированных пользователей")
                            print(" /friends - список друзей")
                            print(" /friend_add - добавить друга")
                            print(" /friend_del - удалить друга")
                            print(" /friend_info - информация о друге")
                            print("ПОЛЕЗНЫЕ КОММАНДЫ:")
                            print(" /gpt - настоящий чат с ChatGPT")
                            print(" /gpt_site - создание сайта по запросу на основе ChatGPT")
                            print(" /wifi - выдаёт список паролей от вайфай, к которым было подключено устройство")
                            print(" /course - курс валют")
                            print(" /block_site - заблокировать/разблокировать сайт")
                            print(" /randpass - создает случайный пароль")
                            print(" /yt - найти видео в ютубе")
                            print(" /sear - найти какую-либо информацию в браузере")
                            print(" /bor - перекинет на сайт, если скучно")
                            print(" /gdz - открыть сайт гдз")
                            print(" /ya - направляет на страницу Yandex")
                            print(" /google - направляет на страницу Google")
                            print(" /trr - перевести текст с русского на английский")
                            print(" /tre - перевести текст с английского на русский")
                            print(" /ip - найти местоположение по ip")
                            print(" /time - узнать время (по МСК)")
                            print(" /timer - установить таймер")
                            print(" /voice - воспроизвести текст в речь")
                            print(" /phone - узнать информацию о номере телефона")
                            print("РАЗВЛЕКАТЕЛЬНЫЕ КОМАНДЫ:")
                            print(" /kazino - игра в казино ")
                            print(" /or - орел или решка?")
                            print(" /spam - спам текстом")
                            print(" /try - правда или неправда")
                            print("МОДЕРАЦИОННЫЕ КОМАНДЫ:")
                            print(" /ac - админ-чат")
                            print(" /give - список того, чего можно выдать")
                            print(" /ban - заблокиравать пользователя")
                            print(" /unban - разблокировать пользователя")
                            print(" /banlist - список заблокированных пользователей")
                            print(" /404 - заблокировать пользователя с удалением его статистики")
                            print(" /deluser - удалить пользователя")
                            print(" /addpromo - добавляет промокод")
                            print(" /delpromo - удаляет промокод")
                            print("СИСТЕМНЫЕ КОММАНДЫ:")
                            print(" /settings - настройки пользования")
                            print(" /scripts - список скриптов")
                            print(" /add_script - добавляет скрипт")
                            print(" /del_script - удаляет скрипт")
                            print(" /load_script - запускает скрипт")
                            print(" /create_script - создать скрипт")
                            print(" /color - изменить цвет текста")
                            print(" /cp - изменить свой пароль")
                            print(" /renick - изменить свой никнейм")
                            print(" /reload - перезагружает программу")
                            print(" /clear - отчистить чат, введённые команды и т.п.")
                            print(" /logout - выйти из аккаунта")
                            print(" /exit - выход из программы")
                            print("=====================================================")

                        elif command == "/spam":
                            try:
                                kol = int(input("Сколько вы хотите строк спама: "))
                                if kol > 0:
                                    text = input("Введите текст: ")
                                    speed = float(input("Введите время задержки между строками: "))
                                    for i in range(kol):
                                        print(text)
                                        time.sleep(speed)
                                else:
                                    print("[!] Количество строк не должно быть менее 1")
                            except ValueError:
                                print("[!] Введено неверное значение")
                        
                        elif command == "/color":
                            color = input("Выберете цвет теста (пишите цифру):\n зелёный(1) \n красный(2) \n синий(3) \n жёлтый(4) \n фиолетовый(5) \n белый(6)\n>>> ")
                            if color == "1":
                                print(Fore.GREEN + '[/] Вы изменили цвет текста на зелёный')
                            elif color == "2":
                                print(Fore.RED + '[/] Вы изменили цвет текста на красный')
                            elif color == "3":
                                print(Fore.BLUE + '[/] Вы изменили цвет текста на синий')
                            elif color == "4":
                                print(Fore.YELLOW + '[/] Вы изменили цвет текста на жёлтый')
                            elif color == "5":
                                print(Fore.MAGENTA + '[/] Вы изменили цвет текста на фиолетовый')
                            elif color == "6":
                                print(Fore.WHITE + '[/] Вы изменили цвет текста на белый')
                            elif color == "/ex":
                                print("[/] Команда прервана")
                            else:
                                print("[!] Введено неверное значение")

                        elif command == "/bor":
                            webbrowser.open_new('https://mneskuchno.com/')

                        elif command == "/ac":
                            if fd['user'][id-1]['privileg'] == privileg[4] or fd['user'][id-1]['privileg'] == privileg[3]:
                                print("Чтобы выйти из админ-чата напишите /ex")
                                while True:
                                    ac_text = input(">>> ")
                                    if ac_text == "/ex":
                                        break
                                    print(f"|Админ-чат| {fd['user'][id-1]['privileg']} {nick} ({id}) -> {ac_text}")
                            else:
                                print("[!] У вас нет прав, чтобы писать в чат администрации")

                        elif command == "/relist":
                            mylist = input("Введите запись: ")
                            addlist(mylist)
                            print("[/] Запись успешно добавлена!")

                        elif command == "/list":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            print(fd["user"][id-1]["list"])

                        elif command == "/dellist":
                            mylist = ""
                            addlist(mylist)
                            print("[/] Запись успешно удалена")
                    
                        elif command == "/reload":
                            file_path = r'C:\Users\Maks\OneDrive\Рабочий стол\Files\программирование\Python\Manager'
                            os.system("start "+file_path)
                            sys.exit("Перезагрузка...")
                        
                        elif command == "/randpass":
                            def password():
                                symbol_password = ['1','2','3','4','5','6','7','8','9','0','q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m','Q','W','E','R','T','Y','U','I','O','P','A','S','D','F','G','H','J','K','L','Z','X','C','V','B','N','M']
                                lenght = int(input('Введите длину пароля: '))
                                password = ''
                                for i in range(lenght):
                                    password += random.choice(symbol_password)
                                print("[/] Ваш пароль: " + password)
                            password()
                                            
                        elif command == "/clear":
                            os.system("cls")
                            print(f"{nick} отчистил(а) чат")

                        elif command == "/yt":
                            search = input("Что найти (/ex для отмены): ")
                            if search == "/ex":
                                print("[/] Отменено")
                            else:
                                webbrowser.open(f'https://www.youtube.com/search?q={search}&oq={search}')

                        elif command == "/sear":
                            search = input("Что найти (/ex для отмены): ")
                            if search == "/ex":
                                print("[/] Отменено")
                            else:
                                webbrowser.open(f'https://www.google.com/search?q={search}&oq={search}')

                        elif command == "/exit":
                            print("Спасибо, что зашли! До встречи!")
                            pygame.mixer.music.load("sounds/close.mp3")
                            pygame.mixer.music.play()
                            time.sleep(1)
                            sys.exit()

                        elif command == "/gdz":
                            webbrowser.open_new('https://gdz.ru/')
                            print("[/] Открыто")
                        
                        elif command == "/info":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            friends_count = 0
                            for friend in fd['user'][id-1]['friends']:
                                friends_count += 1
                            print("\n================== ИНФОРМАЦИЯ ==================")
                            print(f"Ник: {nick}")
                            print(f"Id: {id}")
                            if friends_count == 0:
                                print("Друзей: нет")
                            else:
                                print(f"Друзей: {friends_count}")
                            print("Баланс:", fd["user"][id-1]["balance"])
                            print(f"Титул: {fd['user'][id-1]['title']}")
                            print(f"Привилегия: {fd['user'][id-1]['privileg']}")
                            print(f"Репутация: {fd['user'][id-1]['reputation']}")
                            print("Введено команд:", fd["user"][id-1]["comm"])
                            print("==================================================\n")

                        elif command == "/balance" or command == "/bal":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            bal = fd['user'][id-1]['balance']
                            print(f"[/] Ваш баланс: {bal}")

                        elif command == "/pay":
                            try:
                                with open('data.json', 'r', encoding='utf-8') as jf:
                                    fd = json.load(jf)
                                pay = int(input("Сколько хотите передать денег: "))
                                if pay > fd["user"][id-1]["balance"]:
                                    print("[!] У вас недостаточно денег")
                                elif pay < 1:
                                    print("[!] Введено неверное значение")
                                else:
                                    try:
                                        pay_nick = int(input("Получатель (id): "))
                                        temp_maxid = 0
                                        for temp_balance_user_for1 in fd["id"]:
                                            temp_maxid += 1
                                        if pay_nick > temp_maxid:
                                            print("[!] Такого пользователя не существует")
                                        temp_user = 0
                                        for pay_nick_temp in fd["id"]:
                                            if fd["user"][temp_user].get("id") == pay_nick:
                                                if fd["user"][temp_user].get("id") == id:
                                                    print("[!] Самому себе переводить деньги нельзя")
                                                else:
                                                    fd["user"][temp_user]["balance"] += pay
                                                    fd["user"][id-1]["balance"] -= pay
                                                    with open('data.json', 'w') as jf:
                                                        json.dump(fd, jf, indent=4)
                                                    print("[/] Вы успешно перевели деньги")
                                                break
                                            temp_user += 1                               
                                    except ValueError:
                                        print("[!] Вводите id пользователя")
                            except ValueError:
                                print("[!] Введено неверное значение")    
                    
                        elif command == "/balance_user":
                            try:
                                with open('data.json', 'r', encoding='utf-8') as jf:
                                    fd = json.load(jf)
                                balance_user_id = int(input("Введите id пользователя: "))
                                temp_maxid = 0
                                for temp_balance_user_for1 in fd["id"]:
                                    temp_maxid += 1
                                if balance_user_id > temp_maxid:
                                    print("[!] Такого пользователя не существует")
                                else:
                                    temp_balance_user = 0
                                    for temp_balance_user_for2 in fd["id"]:
                                        if fd["user"][temp_balance_user].get("id") == balance_user_id:
                                            balance_user_nick = fd["user"][temp_balance_user]["name"]
                                            balance_user = fd["user"][temp_balance_user]["balance"]
                                            print(f"Баланс {balance_user_nick} ({balance_user_id}): {balance_user}")
                                            break
                                        temp_balance_user += 1   
                            except ValueError:
                                print("[!] Вводите id пользователя")
                        
                        elif command == "/balancetop":
                            print("\n=========== ТОП 10 БОГАТЫХ ПОЛЬЗОВАТЕЛЕЙ ===========")
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            top_3 = sorted(fd["user"], key=lambda x: x["balance"], reverse=True)[:10]
                            mesto = 1
                            for item in top_3:
                                print(f"{mesto}. {item['name']} - {item['balance']}")   
                                mesto += 1
                            print("\n====================================================")

                        elif command == "/ban":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            if fd['user'][id-1]['privileg'] == privileg[3] or fd['user'][id-1]['privileg'] == privileg[4]:
                                try:
                                    ban_nick = int(input("Кого заблокировать (id) (/ex для отмены): "))
                                    reason = input("Причина (/ex для отмены): ")
                                    if ban_nick == "/ex" or reason == "/ex":
                                        print("[/] Отменено")
                                    else:
                                        with open('data.json', 'r+', encoding='utf-8') as jf:
                                            fd = json.load(jf)
                                        user_in_json = False
                                        user_id_ban = 0
                                        for user in fd['user']:
                                            if user['id'] == ban_nick:
                                                user_in_json = True
                                                break
                                            user_id_ban += 1
                                        if user_in_json:
                                            if fd['user'][id-1]['privileg'] == privileg[3] and fd['user'][ban_nick-1]['privileg'] == privileg[4] or fd['user'][ban_nick-1]['privileg'] == privileg[3]:
                                                print("[!] Вы не можете заблокировать вышестоящего пользователя и модератора")
                                            else:
                                                user_in_json = False
                                                user_id_ban = 0
                                                for user in fd['user']:
                                                    if user['id'] == ban_nick:
                                                        user_in_json = True
                                                        break
                                                    user_id_ban += 1
                                                if user_in_json:
                                                    fd['user'][user_id_ban]['ban'] = True
                                                    fd['user'][user_id_ban]['who_ban'] = nick
                                                    if reason == "":
                                                        reason = "нет причины"
                                                    else:
                                                        fd['user'][user_id_ban]['reason'] = reason
                                                    with open('data.json', 'w') as jf:
                                                        json.dump(fd, jf, indent=4)
                                                    download_ban_nick = 0
                                                    for user in fd['user']:
                                                        if user['id'] == ban_nick:
                                                            ban_nick_name = fd['user'][download_ban_nick]['name']
                                                            break
                                                        download_ban_nick += 1
                                                    if ban_nick == fd['user'][id-1]['id']:
                                                        with open('settings.json', 'r', encoding='utf-8') as jf:
                                                            settings = json.load(jf)
                                                        settings['session']['name'] = ""
                                                        settings['session']['password'] = ""
                                                        with open('settings.json', 'w') as jf:
                                                            json.dump(settings, jf, indent=4)
                                                    print(f"==== Блокировка аккаунта выдана успешно ====")
                                                    print(f"Заблокирован: {ban_nick_name} ({ban_nick})")
                                                    print(f"Заблокировал: {nick} ({id})")
                                                    print(f"Причина: {reason}")
                                                    print(f"Время: {datetime.datetime.now().strftime('%D %H:%M:%S')}")
                                                    print("=============================================")
                                                else:
                                                    print("[!] Пользователь не найден")
                                        else:
                                            print("[!] Пользователь не найден")
                                except ValueError:
                                    print("[!] Вводите id пользователя")
                            else:
                                print("[!] Вы не можете использовать эту команду")

                        elif command == "/unban":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            if fd['user'][id-1]['privileg'] == privileg[3] or fd['user'][id-1]['privileg'] == privileg[4]:
                                try:
                                    unban_nick = int(input("Кого разблокировать (id) (/ex для отмены): "))
                                    if unban_nick == "/ex":
                                        print("[/] Отменено")
                                    else:
                                        with open('data.json', 'r+', encoding='utf-8') as jf:
                                            fd = json.load(jf)
                                        user_in_json = False
                                        user_id_unban = 0
                                        for user in fd['user']:
                                            if user['id'] == unban_nick:
                                                user_in_json = True
                                                break
                                            user_id_unban += 1
                                        if user_in_json:
                                            if fd['user'][id-1]['privileg'] == privileg[3] and fd['user'][unban_nick-1]['privileg'] == privileg[4] or fd['user'][unban_nick-1]['privileg'] == privileg[3]:
                                                print("[!] Вы не можете разблокировать вышестоящего пользователя и модератора")
                                            else:
                                                user_in_json = False
                                                user_id_unban = 0
                                                for user in fd['user']:
                                                    if user['id'] == unban_nick:
                                                        user_in_json = True
                                                        break
                                                    user_id_unban += 1
                                                if user_in_json:
                                                    if fd['user'][user_id_unban]['ban'] is True or fd['user'][user_id_unban]['404'] is True:
                                                        fd['user'][user_id_unban]['ban'] = False
                                                        fd['user'][user_id_unban]['who_ban'] = "No-Name"
                                                        fd['user'][user_id_unban]['reason'] = "нет причины"
                                                        fd['user'][user_id_unban]['404'] = False
                                                        with open('data.json', 'w') as jf:
                                                            json.dump(fd, jf, indent=4)
                                                        download_unban_nick = 0
                                                        for user in fd['user']:
                                                            if user['id'] == unban_nick:
                                                                unban_nick_name = fd['user'][download_unban_nick]['name']
                                                                break
                                                            download_unban_nick += 1     
                                                        print("===== Разблокировка успешная =====")
                                                        print(f"Разблокирован: {unban_nick_name} ({unban_nick})")
                                                        print(f"Разблокировал: {nick} ({id})")
                                                        print(f"Время: {datetime.datetime.now().strftime('%D %H:%M:%S')}")
                                                        print("==================================")
                                                    else:
                                                        print("[!] Пользователь не заблокирован")
                                                else:
                                                    print("[!] Пользователь не найден")
                                        else:
                                            print("[!] Пользователь не найден")
                                except ValueError:
                                    print("[!] Вводите id пользователя")
                            else:
                                print("[!] Вы не можете использовать эту команду")

                        elif command == "/404":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            if fd['user'][id-1]['privileg'] == privileg[4]:
                                try:
                                    nick_404 = int(input("Кого разблокировать (id) (/ex для отмены): "))
                                    if nick_404 == "/ex":
                                        print("[/] Отменено")
                                    else:
                                        with open('data.json', 'r', encoding='utf-8') as jf:
                                            fd = json.load(jf)
                                        user_in_json = False
                                        user_id_404 = 0
                                        for user in fd['user']:
                                            if user['id'] == nick_404:
                                                user_in_json = True
                                                break
                                            user_id_404 += 1
                                        if user_in_json:
                                            fd['user'][user_id_404]['ban'] = False
                                            fd['user'][user_id_404]['friends'] = []
                                            fd['user'][user_id_404]['friends_request'] = []
                                            fd['user'][user_id_404]['friends_your_request'] = []
                                            fd['user'][user_id_404]['reason'] = "нет причины"
                                            fd['user'][user_id_404]['who_ban'] = "No-Name"
                                            fd['user'][user_id_404]['balance'] = 10000
                                            fd['user'][user_id_404]['use_promocode'] = False
                                            fd['user'][user_id_404]['privileg'] = "Пользователь"
                                            fd['user'][user_id_404]['comm'] = 0
                                            fd['user'][user_id_404]['reputation'] = 0
                                            fd['user'][user_id_404]['list'] = ""
                                            fd['user'][user_id_404]['title'] = "Я пользователь данной программы. Давай дружить"
                                            fd['user'][user_id_404]['404'] = True
                                            with open('data.json', 'w') as jf:
                                                json.dump(fd, jf, indent=4)
                                            download_nick_404 = 0
                                            for user in fd['user']:
                                                if user['id'] == nick_404:
                                                    nick_404_name = fd['user'][download_nick_404]['name']
                                                    break
                                                download_nick_404 += 1
                                            reason_404 = "404"
                                            if nick_404 == fd['user'][id-1]['id']:
                                                with open('settings.json', 'r', encoding='utf-8') as jf:
                                                    settings = json.load(jf)
                                                settings['session']['name'] = ""
                                                settings['session']['password'] = ""
                                                with open('settings.json', 'w') as jf:
                                                    json.dump(settings, jf, indent=4)
                                            print("===== Успешно заблокирован 404 =====")
                                            print(f"Заблокирован: {nick_404_name} ({nick_404})")
                                            print(f"Заблокировал: {nick} ({id})")
                                            print(f"Время: {datetime.datetime.now().strftime('%D %H:%M:%S')}")
                                            print("====================================")
                                        else:
                                            print("[!] Пользователь не найден")
                                except ValueError:
                                    print("[!] Вводите id пользователя")
                            else:
                                print("[!] Вы не можете использовать эту команду")
                                
                        elif command == "/banlist":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            print("=================== БАНЫ ===================")
                            for user in fd['user']:
                                if user['ban'] == True:
                                    print(f"{user['who_ban']} заблокировал {user['name']}. Причина: {user['reason']}")
                                elif user['404'] == True:
                                    print(f"Создатель заблокировал {user['name']}. Причина: 404")
                            print("============================================")

                        elif command == "/kazino":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            numbers = '1234567890'
                            try:
                                stavka = int(input("Ваша ставка: "))
                                if stavka < 1000:
                                    print("[!] Ставка не должна быть меньше 1000")
                                else:
                                    if fd['user'][id-1]['balance'] < stavka:
                                        print("[!] У вас недостаточно денег, чтобы поставить такую ставку")
                                    else:
                                        fd['user'][id-1]['balance'] -= stavka
                                        def getPlus():
                                            global r1, r2, r3
                                            r1 = random.choice(numbers)
                                            r2 = random.choice(numbers)
                                            r3 = random.choice(numbers)
                                            print("Идёт подбор числа: ", r1, end='')
                                            print(r2, end='')
                                            print(r3, end='')
                                            time.sleep(0.09)
                                            print('\r', end='')
                                            print('\r', end='')
                                            print('\r', end='')
                                        for i in range(50):
                                            getPlus()
                                        rand_numbers = r1 + r2 + r3
                                        win_numbers = ["000","111", "222", "333", "444", "555", "666", "777", "888", "999", "123", "100", "200", "300", "400", "500", "600", "700", "800", "900"]
                                        if rand_numbers in win_numbers:
                                            stavka *= 5
                                            print("Поздравляем, вам выпали числа " + rand_numbers + "! Вы выиграли", stavka, "$")
                                            with open('data.json', 'r+', encoding='utf-8') as jf:
                                                fd = json.load(jf)
                                            fd['user'][id-1]['balance'] += stavka
                                            with open('data.json', 'w') as jf:
                                                json.dump(fd, jf, indent=4)
                                        else:
                                            with open('data.json', 'r+', encoding='utf-8') as jf:
                                                fd = json.load(jf)
                                            fd['user'][id-1]['balance'] -= stavka
                                            with open('data.json', 'w') as jf:
                                                json.dump(fd, jf, indent=4)
                                            print("[/] Вы ничего не выиграли. Выпали числа " + rand_numbers)
                            except ValueError:
                                print("[!] Введено неверное значене")

                        elif command == "/or":
                            try:
                                stavka = int(input("Ваша ставка (/ex для отмены): "))
                                if stavka == "/ex":
                                    print("[/] Отменено")
                                else:
                                    with open('data.json', 'r+', encoding='utf-8') as jf:
                                        fd = json.load(jf)
                                    if fd['user'][id-1]['balance'] < stavka:
                                        print("[!] У вас недостаточно денег, чтобы поставить такую ставку")
                                    elif stavka < 10:
                                        print("[!] Ставка не должна быть меньше 10")
                                    else:
                                        rand = input("Выберете: орел (1), решка (2) (/ex для отмены): ")
                                        if rand == "/ex":
                                            print("[/] Отменено")
                                        elif rand == "1" or rand == "2":
                                            print("Монетка подкинута...")
                                            time.sleep(2)
                                            r = random.choice(["1", "2"])
                                            if rand == r:
                                                stavka *= 2
                                                if r == "1":
                                                    print("[/] Выпал Орёл. Вы выиграли", stavka)
                                                else:
                                                    print("[/] Выпала Решкаю Вы выиграли", stavka)
                                                with open('data.json', 'r+', encoding='utf-8') as jf:
                                                    fd = json.load(jf)
                                                fd['user'][id-1]['balance'] += stavka
                                                with open('data.json', 'w') as jf:
                                                    json.dump(fd, jf, indent=4)
                                            else:
                                                with open('data.json', 'r+', encoding='utf-8') as jf:
                                                    fd = json.load(jf)
                                                fd['user'][id-1]['balance'] -= stavka
                                                with open('data.json', 'w') as jf:
                                                    json.dump(fd, jf, indent=4)
                                                print("[/] Вы проиграли")
                                        else:
                                            print("[!] Введено неверное значение")
                            except ValueError:
                                print("[!] Введено неверное значение")

                        elif command == "/update":
                            updates = ["добавлена система регистрации, входа, id, банов, промокодов, денег, привилегий и т.п.", "возможность изменить титул, заметку, выдать / убрать репутацию, изменить никнейм и т.п.", "счетчик введённых команд", "хранение данных пользователя", "переделаны множество сообщений", "исправлены баги", "возможность модерирования"]
                            print("Информация последнего обновления (1.5):")
                            for i in updates:
                                print(f" - {i}")

                        elif command == "/donate":
                            print("Привилегии, которые есть в этой прграмме:")
                            for i in privileg:
                                print(f" - {i}")
                        
                        elif command == "/give":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            if fd['user'][id-1]['privileg'] == privileg[3] or fd['user'][id-1]['privileg'] == privileg[4]:
                                print("\nКоманды для выдачи:")
                                print(" /give_balance - выдаёт деньги")
                                print(" /give_rep - выдаёт репутацию")
                                print(" /give_perm - выдаёт привилегию\n")
                            else:
                                print("[!] Вы не можете использовать эту команду")
                        
                        elif command == "/give_balance":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            if fd['user'][id-1]['privileg'] == privileg[3] or fd['user'][id-1]['privileg'] == privileg[4]:
                                try:
                                    add_balance_user = int(input("Кому выдать деньги (id) (/ex для отмены): "))
                                    amount = int(input("Сумма перевода (/ex для отмены): "))
                                    if add_balance_user == "/ex":
                                        print("[/] Отменено")
                                    else:
                                        add_balance_user_checked = False
                                        add_balance_user_id = 0
                                        for user in fd['user']:
                                            if user['id'] == add_balance_user:
                                                add_balance_user_checked = True
                                                break
                                            add_balance_user_id += 1
                                        if add_balance_user_checked:
                                            add_balance_nick = fd['user'][add_balance_user_id]['name']
                                            fd['user'][add_balance_user_id]['balance'] += amount
                                            with open('data.json', 'w') as jf:
                                                json.dump(fd, jf, indent=4)
                                            print(f"[/] Перевод пользователю {add_balance_nick} ({add_balance_user_id+1}) успешно выполнен в размене {amount}$")
                                        else:
                                            print("[!] Пользователь не найден")
                                except ValueError:
                                    print("[!] Введено неверное значение")
                            else:
                                print("[!] Вы не можете использовать эту команду")

                        elif command == "/give_rep":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            if fd['user'][id-1]['privileg'] == privileg[3] or fd['user'][id-1]['privileg'] == privileg[4]:
                                try:
                                    add_rep_user = int(input("Кому выдать репутацию (id) (/ex для отмены): "))
                                    amount = int(input("Количество репутации (/ex для отмены): "))
                                    if add_rep_user == "/ex":
                                        print("[/] Отменено")
                                    else:
                                        add_rep_user_checked = False
                                        add_rep_user_id = 0
                                        for user in fd['user']:
                                            if user['id'] == add_rep_user:
                                                add_rep_user_checked = True
                                                break
                                            add_rep_user_id += 1
                                        if add_rep_user_checked:
                                            add_rep_nick = fd['user'][add_rep_user_id]['name']
                                            fd['user'][add_rep_user_id]['reputation'] += amount
                                            with open('data.json', 'w') as jf:
                                                json.dump(fd, jf, indent=4)
                                            print(f"[/] Вы выдали {amount} репутаций пользователю {add_rep_nick} ({add_rep_user_id+1})")
                                        else:
                                            print("[!] Пользователь не найден")
                                except ValueError:
                                    print("[!] Введено неверное значение")
                            else:
                                print("[!] Вы не можете использовать эту команду")

                        elif command == "/give_perm":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            if fd['user'][id-1]['privileg'] == privileg[4]:
                                try:
                                    add_perm_user = int(input("Кому выдать привилегию (id) (/ex для отмены): "))
                                    perm = input("Какую привилегию (/ex для отмены): ")
                                    if add_perm_user == "/ex":
                                        print("[/] Отменено")
                                    else:
                                        if perm in privileg:
                                            add_perm_user_checked = False
                                            add_perm_user_id = 0
                                            for user in fd['user']:
                                                if user['id'] == add_perm_user:
                                                    add_perm_user_checked = True
                                                    break
                                                add_perm_user_id += 1
                                            if add_perm_user_checked:
                                                add_perm_nick = fd['user'][add_perm_user_id]['name']
                                                fd['user'][add_perm_user_id]['privileg'] = perm
                                                with open('data.json', 'w') as jf:
                                                    json.dump(fd, jf, indent=4)
                                                print(f"[/] Вы выдали пользователю {add_perm_nick} ({add_perm_user_id+1}) привилегию {perm}")
                                            else:
                                                print("[!] Пользователь не найден")
                                        else:
                                            print("[!] Такой привилегии не существует. Список доступных: /donate")
                                except ValueError:
                                    print("[!] Введено неверное значение")
                            else:
                                print("[!] Вы не можете использовать эту команду")

                        elif command == "/promo":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            if fd['user'][id-1]['use_promocode'] is False:
                                promocode = input("Введите промокод (/ex для отмены): ")
                                if promocode == "/ex":
                                    print("[/] Отменено")
                                else:
                                    id_prom = 0
                                    promocode_checked = False
                                    for prom in fd['promocode']:
                                        if fd['promocode'][id_prom]['name'] == promocode:
                                            promocode_checked = True
                                            break
                                        id_prom += 1
                                    if promocode_checked:
                                        fd['user'][id-1]['balance'] += fd['promocode'][id_prom]['money']
                                        fd['user'][id-1]['use_promocode'] = True
                                        print(f"[/] Вы использоватли промокод {promocode}. Вы получили {fd['promocode'][id_prom]['money']} $")
                                        with open('data.json', 'w') as jf:
                                            json.dump(fd, jf, indent=4)
                                    else:
                                        print("[!] Такого промокода не существует")
                            else:
                                print("[!] Вы уже активировали промокод")

                        elif command == "/addpromo":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            if fd['user'][id-1]['privileg'] == privileg[4]:
                                name_promo = input("Введите название промокода (/ex для отмены): ")
                                if name_promo == "/ex":
                                    print("[/] Отменено")
                                else:
                                    try:
                                        money_promo = int(input("Сумма при получении (/ex для отмены): "))
                                        if money_promo == "/ex":
                                            print("[/] Отменено")
                                        else:
                                            prom_id = 0
                                            prom_check = False
                                            for prom in fd['promocode']:
                                                if fd['promocode'][prom_id]['name'] == name_promo:
                                                    prom_check = True
                                                    break
                                                prom_id += 1
                                            if prom_check is False:
                                                add_prom = {
                                                    "name": name_promo,
                                                    "money": money_promo
                                                }
                                                with open('data.json','r+', encoding='utf-8') as jf:
                                                    fd = json.load(jf)
                                                    fd["promocode"].append(add_prom)
                                                    jf.seek(0)
                                                    json.dump(fd, jf, indent = 4)
                                                print(f"[/] Вы успешно добавили промокод {name_promo} с призом {money_promo} $")
                                            else:
                                                print("[!] Такой промокод уже имеется")
                                        
                                    except ValueError:
                                        print("[!] Введено неверное значение")

                        elif command == "/delpromo":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            if fd['user'][id-1]['privileg'] == privileg[4]:
                                name_promo = input("Введите название промокода (/ex для отмены): ")
                                if name_promo == "/ex":
                                    print("[/] Отменено")
                                else:
                                    prom_id = 0
                                    prom_check = False
                                    for prom in fd['promocode']:
                                        if fd['promocode'][prom_id]['name'] == name_promo:
                                            prom_check = True
                                            break
                                        prom_id += 1
                                    if prom_check is True:
                                        fd['promocode'].pop(prom_id)
                                        with open('data.json', 'w') as jf:
                                            json.dump(fd, jf, indent=4)
                                        print(f"[/] Вы успешно удалили промокод {name_promo}")
                                    else:
                                        print("[!] Промокод не найден")                                
                            else:
                                print("[!] Вы не можете использовать эту команду")

                        elif command == "/renick":
                            new_nick = input("Введите новый никнейм: ")
                            if new_nick == "" or " " in new_nick or "/" in new_nick or "{" in new_nick or "}" in new_nick:
                                print("[!] Никнейм содержит запрещённые символы")
                            else:
                                if len(new_nick) < 3 or len(new_nick) > 15:
                                    print("[!] Никнейм не должен быть менее 16 символов и более 2 символов")
                                else:
                                    renick(new_nick)
                                    with open('data.json', 'r', encoding='utf-8') as jf:
                                        fd = json.load(jf)
                                        nick = fd["user"][id-1]["name"]
                                    with open('settings.json', 'r', encoding='utf-8') as jf:
                                        settings = json.load(jf)
                                    settings["session"]["name"] = ""
                                    settings["session"]["password"] = ""
                                    with open('settings.json', 'w') as jf:
                                        json.dump(settings, jf, indent=4)

                        elif command == "/trr":
                            translator = translate.Translator(from_lang="ru", to_lang="en")
                            text_ru = input("Введите текст на русском: ")
                            print("[/] Перевод: ", translator.translate(text_ru))

                        elif command == "/tre":
                            translator = translate.Translator(from_lang="en", to_lang="ru")
                            text_en = input("Введите текст на английском: ")
                            print("[/] Перевод: ", translator.translate(text_en))
                        
                        elif command == "/changepassword" or command == "/cp":
                            new_password = input("Введите новый пароль (/ex для отмены): ")
                            if new_password == "/ex":
                                print("[/] Отменено")
                            else:
                                if new_password == "" or " " in new_password:
                                    print("[!] Пароль содержит запрещённые символы")
                                else:
                                    changepassword(new_password)
                        
                        elif command == "/retitle":
                            new_title = input("Введите новый титул (/ex для отмены): ")
                            if new_title == "/ex":
                                print("[/] Отменено")
                            else:
                                retitle(new_title)

                        elif command == "/try":
                            ask_try = input("Введите удтверждение: ")
                            rand_try = ["ДА", "НЕТ"]
                            print(f"[/] {random.choice(rand_try)}")
                        
                        elif command == "/logout":
                            with open('settings.json', 'r', encoding='utf-8') as jf:
                                settings = json.load(jf)
                            settings['session']['name'] = ""
                            settings['session']['password'] = ""
                            with open('settings.json', 'w') as jf:
                                json.dump(settings, jf, indent=4)
                            return login()
                        
                        elif command == "/ya":
                            webbrowser.open(f'https://ya.ru/')
                            print("[/] Открыто")
                        
                        elif command == "/google":
                            webbrowser.open(f'https://www.google.com')
                            print("[/] Открыто")

                        elif command == "/deluser":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            if fd['user'][id-1]['privileg'] == privileg[4]:
                                try:
                                    del_user = int(input("Кого удатить (id) (/ex для отмены): "))
                                    if del_user == "/ex":
                                        print("[/] Отменено")
                                    else:
                                        with open('data.json', 'r', encoding='utf-8') as jf:
                                            fd = json.load(jf)
                                        user_stat = 0
                                        user_checked = False
                                        for user in fd['user']:
                                            if user['id'] == del_user:
                                                nick_user = fd['user'][user_stat]['name']
                                                user_checked = True
                                                break
                                            user_stat += 1
                                        if user_checked:
                                            if del_user == fd['user'][id-1]['id']:
                                                with open('settings.json', 'r', encoding='utf-8') as jf:
                                                    settings = json.load(jf)
                                                settings['session']['name'] = ""
                                                settings['session']['password'] = ""
                                                with open('settings.json', 'w') as jf:
                                                    json.dump(settings, jf, indent=4)
                                            fd['user'] = [item1 for item1 in fd['user'] if item1['id'] != del_user]
                                            fd['id'].pop(del_user-1)
                                            with open('data.json', 'w') as jf:
                                                json.dump(fd, jf, indent=4)
                                            with open('data.json', 'r', encoding='utf-8') as jf:
                                                fd = json.load(jf)
                                            start_id = 1
                                            for user in fd['user']:
                                                with open('data.json', 'r', encoding='utf-8') as jf:
                                                    fd = json.load(jf)
                                                if fd['user'][start_id-1]['id'] != start_id:
                                                    fd['user'][start_id-1]['id'] = start_id
                                                    with open('data.json', 'w') as jf:
                                                        json.dump(fd, jf, indent=4)
                                                    with open('data.json', 'r', encoding='utf-8') as jf:
                                                        fd = json.load(jf)
                                                    for user_id in fd['id']:
                                                        temp_nick_for_deluser = fd['user'][start_id-1]['name']
                                                        if fd['id'][start_id-1][temp_nick_for_deluser]['id'] != start_id:
                                                            fd['id'][start_id-1][temp_nick_for_deluser]['id'] = start_id
                                                        with open('data.json', 'w') as jf:
                                                            json.dump(fd, jf, indent=4)
                                                start_id += 1
                                        else:
                                            print("[!] Пользователь не найден")
                                except ValueError:
                                    print("[!] Введено неверное значение")
                            else:
                                print("[!] Вы не можете использовать эту команду")
                                
                        elif command == "/ip":
                            def get_info_by_ip(ip='127.0.0.1'):
                                try:
                                    response = requests.get(url=f'http://ip-api.com/json/{ip}').json()                                
                                    data = {
                                        '[IP]': response.get('query'),
                                        '[Int prov]': response.get('isp'),
                                        '[Org]': response.get('org'),
                                        '[Country]': response.get('country'),
                                        '[Region Name]': response.get('regionName'),
                                        '[City]': response.get('city'),
                                        '[ZIP]': response.get('zip'),
                                        '[Lat]': response.get('lat'),
                                        '[Lon]': response.get('lon'),
                                    }
                                    print(f"[/] Информация об ip-адресе ({ip}):")
                                    for k, v in data.items():
                                        print(f'{k}: {v}')
                                    area = folium.Map(location=[response.get('lat'), response.get('lon')])
                                    area.save(f'{response.get("query")}_{response.get("city")}.html')
                                except requests.exceptions.ConnectionError:
                                    print('[!] Проверте подключение к интернету')
                            ip = input('Введите ip: ')
                            get_info_by_ip(ip=ip)

                        elif command == "/rep":
                            try:
                                add_rep = int(input("Кому выдать репутацию (id) (/ex для отмены): "))
                                if add_rep == "/ex":
                                    print("[/] Отменено")
                                else:
                                    with open('data.json', 'r', encoding='utf-8') as jf:
                                        fd = json.load(jf)
                                    user_checked = False
                                    user_id = 0
                                    for user in fd['user']:
                                        if fd['user'][user_id]['id'] == add_rep:
                                            user_checked = True
                                            break
                                        user_id += 1
                                    if user_checked:
                                        fd['user'][user_id]['reputation'] += 1
                                        with open('data.json', 'w') as jf:
                                            json.dump(fd, jf, indent=4)
                                        print(f"[/] Репутация пользователя {fd['user'][user_id]['name']} повышена на 1")
                                    else:
                                        print("[!] Пользователь не найден")
                            except ValueError:
                                print("[!] Введено неверное значение")
                        
                        elif command == "/unrep":
                            try:
                                del_rep = int(input("Кому понизить репутацию (id) (/ex для отмены): "))
                                if del_rep == "/ex":
                                    print("[/] Отменено")
                                else:
                                    with open('data.json', 'r', encoding='utf-8') as jf:
                                        fd = json.load(jf)
                                    user_checked = False
                                    user_id = 0
                                    for user in fd['user']:
                                        if fd['user'][user_id]['id'] == del_rep:
                                            user_checked = True
                                            break
                                        user_id += 1
                                    if user_checked:
                                        fd['user'][user_id]['reputation'] -= 1
                                        with open('data.json', 'w') as jf:
                                            json.dump(fd, jf, indent=4)
                                        print(f"[/] Репутация пользователя {fd['user'][user_id]['name']} понижена на 1")
                                    else:
                                        print("[!] Пользователь не найден")
                            except ValueError:
                                print("[!] Введено неверное значение")
                        
                        elif command == "/user":
                            try:
                                user = int(input("Кого найти (/ex для отмены): "))
                                if user == "/ex":
                                    print("[/] Отменено")
                                else:
                                    with open('data.json', 'r', encoding='utf-8') as jf:
                                        fd = json.load(jf)
                                    user_checked = False
                                    user_id = 0
                                    for u in fd['user']:
                                        if fd['user'][user_id]['id'] == user:
                                            user_checked = True
                                            break
                                        user_id += 1
                                    friends_count = 0
                                    for friend in fd['user'][user_id]['friends']:
                                        friends_count += 1
                                    if user_checked:
                                        with open('data.json', 'r', encoding='utf-8') as jf:
                                            fd = json.load(jf)
                                        print("\n=========== ИНФОРМАЦИЯ О ПОЛЬЗОВАТЕЛЕ ===========")
                                        print(f"Ник: {fd['user'][user_id]['name']}")
                                        print(f"Id: {fd['user'][user_id]['id']}")
                                        if friends_count == 0:
                                            print("Друзей: нет")
                                        else:
                                            print(f"Друзей: {friends_count}")
                                        print("Баланс:", fd['user'][user_id]['balance'])
                                        print(f"Титул: {fd['user'][user_id]['title']}")
                                        print(f"Привилегия: {fd['user'][user_id]['privileg']}")
                                        print(f"Репутация: {fd['user'][user_id]['reputation']}")
                                        print(f"Введено команд: {fd['user'][user_id]['comm']}")
                                        print("==================================================\n")
                                    else:
                                        print("[!] Пользователь не найден")
                            except ValueError:
                                print("[!] Введено неверное значение")

                        elif command == "/users":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            registered = 0
                            for user in fd['user']:
                                registered += 1
                            print(f"Зарегистрированных {registered}:")
                            user_id = 0
                            for user in fd['user']:
                                print(f"- {fd['user'][user_id]['name']} ({fd['user'][user_id]['id']})")
                                user_id += 1
                                
                        elif command == "/friends":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            print(f"\n=======================================")
                            if fd['user'][id-1]['friends'] == []:
                                print("У вас нет друзей\n")
                            else:
                                friends_count = 0
                                for friend in fd['user'][id-1]['friends']:
                                    friends_count += 1
                                print(f"Ваши друзья ({friends_count}):")
                                for friend in fd['user'][id-1]['friends']:
                                    print(f" - {fd['user'][friend-1]['name']} ({fd['user'][friend-1]['id']})")
                            if fd['user'][id-1]['friends_request'] == []:
                                print("\nВам не отправляли запросы в друзья\n")
                            else:
                                print("Хотят подружиться:")
                                for friend_request in fd['user'][id-1]['friends_request']:
                                    print(f" - {fd['user'][friend_request-1]['name']} ({fd['user'][friend_request-1]['id']})")
                            if fd['user'][id-1]['friends_your_request'] == []:
                                print("\nНет отправленных запросов в друзья")
                            else:
                                print("Отправлены запросы:")
                                for friend_your_request in fd['user'][id-1]['friends_your_request']:
                                    print(f" - {fd['user'][friend_your_request-1]['name']} ({fd['user'][friend_your_request-1]['id']})")
                            print("=======================================\n")
                                    
                        elif command == "/friend_add":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            try:
                                friend_add = int(input("Кого добавить в друзья (id) (/ex для отмены): "))
                                if friend_add == fd['user'][id-1]['id']:
                                    print("[!] Вы не можете добавить самого себя в друзья")
                                else:
                                    friend_add_in_json1 = False
                                    for friend in fd['user'][id-1]['friends']:
                                        if friend == friend_add:
                                            friend_add_in_json1 = True
                                            break
                                    if friend_add_in_json1 is False:
                                        friend_add_in_req2 = False
                                        for friend in fd['user'][id-1]['friends_your_request']:
                                            if friend == friend_add:
                                                friend_add_in_req2 = True
                                                break
                                        if friend_add_in_req2 is False:
                                            friend_checked = False
                                            user_id = 0
                                            for user in fd['user']:
                                                if fd['user'][user_id]['id'] == friend_add:
                                                    friend_checked = True
                                                    break
                                                user_id += 1
                                            if friend_checked:
                                                with open('data.json', 'r', encoding='utf-8') as jf:
                                                    fd = json.load(jf)
                                                if friend_add in fd['user'][id-1]['friends_request']:
                                                    fd['user'][id-1]['friends'].append(friend_add)
                                                    friends_request = fd['user'][id-1]['friends_request']
                                                    new_friends_request = []
                                                    for friend in friends_request:
                                                        if friend != friend_add:
                                                            new_friends_request.append(friend)
                                                    fd['user'][id-1]['friends_request'] = new_friends_request
                                                    
                                                    fd['user'][friend_add-1]['friends_your_request'].remove(id)
                                                    fd['user'][friend_add-1]['friends'].append(id)
                                                else:
                                                    fd['user'][id-1]['friends_your_request'].append(friend_add)
                                                    fd['user'][friend_add-1]['friends_request'].append(id)
                                                with open('data.json', 'w') as jf:
                                                    json.dump(fd, jf, indent=4)
                                                friend_add_name = fd['user'][friend_add-1]['name']
                                                print(f"[/] Добавлен новый друг {friend_add_name}")
                                            else:
                                                print("[!] Такого пользователя не существует")
                                        else:
                                            print("[!] Вы уже отправили запрос в друзья данному пользователю")
                                    else:
                                        print('[!] Вы уже добавляли данного пользователя')
                            except ValueError:
                                print("[!] Введено неверное значение")
                            
                        elif command == "/friend_del":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            try:
                                friend_del = int(input("Кого удалить из друзей (id) (/ex для отмены): "))
                                if friend_del == fd['user'][id-1]['id']:
                                    print("[!] Вы не можете удалять самого себя")
                                else:
                                    users_checked = False
                                    user_id = 0
                                    for user in fd['user']:
                                        if user['id'] == friend_del:
                                            users_checked = True
                                            break
                                        user_id += 1
                                    if users_checked:
                                        friend_del_in_json = False
                                        for friend in fd['user'][id-1]['friends']:
                                            if friend == friend_del:
                                                friend_del_in_json = True
                                                break
                                        if friend_del_in_json:
                                            with open('data.json', 'r', encoding='utf-8') as jf:
                                                fd = json.load(jf)
                                            fd['user'][id-1]['friends'].remove(friend_del)
                                            fd['user'][friend_del-1]['friends'].remove(id)
                                            friend_del_name = fd['user'][friend_del-1]['name']
                                            print(f"[/] {friend_del_name} больше не ваш друг")
                                            with open('data.json', 'w') as jf:
                                                json.dump(fd, jf, indent=4)
                                        else:
                                            friend_del_in_req = False
                                            for friend in fd['user'][id-1]['friends_request']:
                                                if friend == friend_del:
                                                    friend_del_in_req = True
                                                    break
                                            if friend_del_in_req:
                                                with open('data.json', 'r', encoding='utf-8') as jf:
                                                    fd = json.load(jf)
                                                fd['user'][id-1]['friends_request'].remove(friend_del)
                                                fd['user'][friend_del-1]['friends_your_request'].remove(id)
                                                with open('data.json', 'w') as jf:
                                                    json.dump(fd, jf, indent=4)
                                                friend_del_name = fd['user'][friend_del-1]['name']
                                                print(f"[/] Вы отменили запрос {friend_del_name} в друзья")
                                            else:
                                                friend_del_in_your_req = False
                                                for friend in fd['user'][id-1]['friends_your_request']:
                                                    if friend == friend_del:
                                                        friend_del_in_your_req = True
                                                        break
                                                if friend_del_in_your_req:
                                                    with open('data.json', 'r', encoding='utf-8') as jf:
                                                        fd = json.load(jf)
                                                    fd['user'][id-1]['friends_your_request'].remove(friend_del)
                                                    fd['user'][friend_del-1]['friends_request'].remove(id)
                                                    with open('data.json', 'w') as jf:
                                                        json.dump(fd, jf, indent=4)
                                                    friend_del_name = fd['user'][friend_del-1]['name']
                                                    print(f"[/] Вы отменили запрос в друзья пользователю {friend_del_name}")
                                                else:
                                                    print("[!] Данный пользователь не ваш друг")
                                    else:
                                        print("[!] Данного пользователя не существует")
                            except ValueError:
                                print("[!] Введено неверное значение")
                                
                        elif command == "/friend_info":
                            with open('data.json', 'r', encoding='utf-8') as jf:
                                fd = json.load(jf)
                            try:
                                friend_info = int(input("Кого найти (id) (/ex для отмены): "))
                                if friend_info == "/ex":
                                    print("[/] Отменено")
                                else:
                                    friend_info_checked = False
                                    for friend in fd['user'][id-1]['friends']:
                                        if friend == friend_info:
                                            friend_info_checked = True
                                            break
                                    with open('data.json', 'r', encoding='utf-8') as jf:
                                        fd = json.load(jf)
                                    user_checked = False
                                    user_id = 0
                                    for u in fd['user']:
                                        if fd['user'][user_id]['id'] == friend_info:
                                            user_checked = True
                                            break
                                        user_id += 1
                                        friends_count = 0
                                        for friend in fd['user'][id-1]['friends']:
                                            friends_count += 1
                                    if friend_info_checked:
                                        with open('data.json', 'r', encoding='utf-8') as jf:
                                            fd = json.load(jf)
                                        print("\n=========== ИНФОРМАЦИЯ О ПОЛЬЗОВАТЕЛЕ ===========")
                                        print(f"Ник: {fd['user'][user_id]['name']}")
                                        print(f"Id: {fd['user'][user_id]['id']}")
                                        if friends_count == 0:
                                            print("Друзей: нет")
                                        else:
                                            print(f"Друзей: {friends_count}")
                                        print("Баланс:", fd['user'][user_id]['balance'])
                                        print(f"Титул: {fd['user'][user_id]['title']}")
                                        print(f"Привилегия: {fd['user'][user_id]['privileg']}")
                                        print(f"Репутация: {fd['user'][user_id]['reputation']}")
                                        print(f"Введено команд: {fd['user'][user_id]['comm']}")
                                        print("==================================================\n")
                                    else:
                                        print("[!] Данный пользователь не ваш друг")
                                
                            except ValueError:
                                print("[!] Введено неверное значение")
                                
                        elif command == "/v":
                            print(f"Версия программы: {version}")
                            
                        elif command  == "/settings":
                            print("\n============== НАСТРОЙКИ ==============")
                            print(" /settings_ai_token - изменить токен для ChatGPT")
                            print(" /settings_color - цвет текста")
                            print(" /settings_bgcolor - обводка текста")

                        elif command == "/settings_color":
                            with open('settings.json', 'r', encoding='utf-8') as jf:
                                settings = json.load(jf)
                            colors = ["GREEN", "RED", "BLUE", "YELLOW", "PURPLE", "WHITE", "BLACK"]
                            print("Доступные цвета:")
                            for col in colors:
                                print(f" - {col}")
                            color = input("Выберете цвет теста: ")
                            color_up = color.upper()
                            if color == "/ex":
                                print("[/] Отменено")
                            elif color_up in colors:
                                settings['color'] = color_up
                                with open('settings.json','w', encoding='utf-8') as jf:
                                    json.dump(settings, jf, indent = 4)
                                check_color()
                                print(f"[/] Цвет текста изменён на {color}")
                            else:
                                print("[!] Такого цвета не существует или он не доступен")
                                
                        elif command == "/settings_bgcolor":
                            with open('settings.json', 'r', encoding='utf-8') as jf:
                                settings = json.load(jf)
                            bgcolors = ["GREEN", "RED", "BLUE", "YELLOW", "PURPLE", "WHITE", "BLACK"]
                            print("Доступные цвета:")
                            for col in bgcolors:
                                print(f" - {col}")
                            bgcolor = input("Выберете цвет теста: ")
                            bgcolor_up = bgcolor.upper()
                            if bgcolor == "/ex":
                                print("[/] Отменено")
                            elif bgcolor_up in bgcolors:
                                settings['bgcolor'] = bgcolor_up
                                with open('settings.json','w', encoding='utf-8') as jf:
                                    json.dump(settings, jf, indent = 4)
                                check_bgcolor()
                                print(f"[/] Цвет фона текста изменён на {bgcolor}")
                            else:
                                print("[!] Такого цвета не существует или он не доступен")
                            
                        elif command == "/gpt":
                            with open('settings.json', 'r', encoding='utf-8') as jf:
                                settings = json.load(jf)
                            openai.api_key = settings['openai']['token']
                            def get_openai_response(message):
                                response = openai.Completion.create(
                                    engine='text-davinci-003',
                                    prompt=message,
                                    max_tokens=2900,
                                    n=1,
                                    stop=None,
                                    temperature=0.7
                                )
                                return response.choices[0].text.strip()

                            def main():
                                print("[/] Открыт ChatGPT. Для выхода введите /ex")
                                while True:
                                    user_input = input("Вы: ")
                                    if user_input.lower() == '/ex':
                                        print("[/] Отменено")
                                        break
                                    print("Генерация текста, подождите...")
                                    openai_response = get_openai_response(user_input)
                                    print(f"OpenAI:\n{openai_response}\n")
                            main()
                        
                        elif command == "/gpt_site":
                            with open('settings.json', 'r', encoding='utf-8') as jf:
                                settings = json.load(jf)
                            openai.api_key = settings['openai']['token']
                            def get_openai_response(message):
                                response = openai.Completion.create(
                                    engine='text-davinci-003',
                                    prompt=message,
                                    max_tokens=3000,
                                    n=1,
                                    stop=None,
                                    temperature=0.7
                                )
                                return response.choices[0].text.strip()

                            def main():
                                print("[/] Открыт генератор сайтов на основе ChatGPT. Задавайте вопросы на любые темы. Для выхода введите /ex")
                                while True:
                                    user_input = input("Вы: ")
                                    if user_input.lower() == '/ex':
                                        print("[/] Отменено")
                                        break
                                    print("Генерация сайта, подождите...")
                                    settings["site_counter"] += 1
                                    with open('settings.json','w', encoding='utf-8') as jf:
                                        json.dump(settings, jf, indent = 4)
                                    result_input = f"Сделай одностраничный веб сайт на HTML, CSS и JavaScript с шапкой, кнопками с переходом на блоки, содержимым, картинками и подвалом (всё в одном файле). Сделай дизайн сайта в одном стиле (подбери подходящие цвета. Их может быть несколько, главное чтобы сочетались). Напиши содержимого побольше. Содержимым или темой сайта будет являться следующее: \"{user_input}\""
                                    openai_response = get_openai_response(result_input)
                                    site_counter = settings["site_counter"]
                                    with open(f'sites/site{site_counter}.html', 'w', encoding='utf-8') as file:
                                        file.write(openai_response)
                                    print("[/] Сайт готов! Перейдите в каталог /sites")
                            main()
                        
                        elif command == "/timer":
                            try:
                                if timer_active:
                                    pygame.mixer.music.stop()
                                    print("[/] Таймер удалён")
                                    timer_active = False
                                else:
                                    times = int(input("Через сколько секунд напомнить: "))
                                    timer_active = True
                                    def timer(sec):
                                        while sec > 0:
                                            time.sleep(1)
                                            sec -= 1
                                        if timer_active:
                                            pygame.mixer.music.load("sounds/timer.mp3")
                                            pygame.mixer.music.play(loops=-1)
                                            return print("[/] СРАБОТАЛ ТАЙМЕР. Для выключения введите /timer")
                                    def start_timer(seconds):
                                        t = Thread(target=timer, args=(seconds,))
                                        t.start()
                                    start_timer(times)
                            except ValueError:
                                print("[!] Введено неверное значение")
                        
                        elif command == "/voice":
                            voice = input("Введите текст для воспроизведения (/ex для отмены): ")
                            if voice == "/ex":
                                print("[/] Отменено")
                            else:
                                engine.say(voice)
                                engine.runAndWait()
                                # voice2 = list(voice)
                                # rus_letters = ['А','Б','В','Г','Д','Е','Ё','Ж','З','И','Й','К','Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ь','Ы','Ъ','Э','Ю','Я']
                                # eng_letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
                                # if rus_letters in voice2[0] or rus_letters.lower() in voice2[0]:
                                #     speech = gTTS(voice, lang="ru", slow=False)
                                #     speech.save('loads/speech.mp3')
                                # elif eng_letters in voice2[0] or eng_letters.lower() in voice2[0]:
                                #     speech = gTTS(voice, lang="en", slow=False)
                                #     speech.save('loads/speech.mp3')
                                # else:
                                #     print("[!] Что-то пошло не так с распознанием текста")
                        
                        elif command == "/settings_ai_token":
                            with open('settings.json', 'r', encoding='utf-8') as jf:
                                settings = json.load(jf)
                            new_token = input("Введите токен для ChatGPT (/ex для отмены): ")
                            if new_token == "/ex":
                                print("[/] Отменено")
                            else:
                                settings['openai']['token_2'] = settings['openai']['token']
                                settings['openai']['token'] = new_token
                                with open('settings.json','w', encoding='utf-8') as jf:
                                    json.dump(settings, jf, indent = 4)
                                print("[/] Токен был успешно заменён")
                        
                        elif command == "/time":
                            now = datetime.datetime.now()
                            print(f"Дата: {now.strftime('%d %m %Y')}")
                            print(f"Время: {now.strftime('%H:%M:%S')}")
                        
                        elif command == "/wifi":
                            try:
                                data = subprocess.check_output("netsh wlan show profiles").decode('cp866').split('\n')
                                profiles = [i.split(":")[1][1:-1] for i in data if "Все профили пользователей" in i] 
                                pass_wifi = '' 
                                for i in profiles:
                                    results = subprocess.check_output(['netsh', 'wlan', 'show', 'profile', i, 'key=clear']).decode('cp866').split('\n')
                                    for j in results:
                                        if "Содержимое ключа" in j:
                                            pass_wifi += f"{i} -- {j.split(':')[1][1:-1]}\n"
                                print(pass_wifi)
                            except Exception as ex:
                                print(f'[!] Возникла ошибка! Обратитесь в поддержку. {ex}')
                                
                        elif command == "/course":
                            course = input("Какую валюту отследить:\n EUR\n USD\n/ex для отмены: ")
                            response = requests.get('https://www.cbr-xml-daily.ru/daily_json.js').json()
                            if course == "/ex":
                                print("[/] Отменено")
                            elif course.upper() == "EUR":
                                print(f"Курс евро: {response['Valute']['EUR']['Value']} рублей")
                            elif course.upper() == "USD":
                                print(f"Курс доллара: {response['Valute']['USD']['Value']} рублей")
                            else:
                                print("[!] Неизвестная валюта")
                                
                        elif command == "/block_site":
                            try:
                                path_to_hosts = ''
                                if platform.system() == 'Windows':
                                    path_to_hosts = r'C:\Windows\System32\drivers\etc\hosts' 
                                redirect = '127.0.0.1'
                                websites = []
                                def block():
                                    while True:
                                        site = input('Введите сайт для блокировки: ')
                                        if site == '/ex':
                                            break
                                        websites.append(site)
                                        print(f'[/] Ссылка {site} добавлена')
                                        print('Если вы добавили сайты, для выхода введите "/ex"')

                                    with open(path_to_hosts, 'r+') as file:
                                        content = file.read()
                                        for site in websites:
                                            if site in content:
                                                pass
                                            else:
                                                file.write(f'{redirect} {site}\n')
                                def unblock():
                                    while True:
                                        site = input('Введите сайт для блокировки (/ex для отмены): ')
                                        if site == '/ex':
                                            break
                                        websites.append(site)
                                        print(f'Ссылка {site} добавлена!')
                                        print('Если вы добавили сайты, для выхода введите "/ex"')

                                    with open(path_to_hosts, 'r+') as file:
                                        content = file.readlines()
                                        file.seek(0)
                                        for line in content:
                                            if not any(site in line for site in websites):
                                                file.write(line)
                                        file.truncate()
                                while True:
                                    choosing_action = input('ПОМОЩЬ:\n 1 - Заблокировать сайт/ы\n 2 - Разблокировать сайт/ы\n /ex для отмены\n>>> ')
                                    if choosing_action == '1' or choosing_action == '2' or choosing_action == '/ex':
                                        break
                                    print('[!] Выберите вариант ответа')
                                if choosing_action == '1':
                                    block()
                                elif choosing_action == '2':
                                    unblock()
                                elif choosing_action == '/ex':
                                    print("[/] Отменено")
                                else:
                                    print('[!] Выберите вариант ответа')
                            except PermissionError:
                                print("[!] Команда не доступна. Запустите программу с правами администратора")
                        
                        elif command == "/scripts":
                            print("Добавленные скрипты:")
                            with open('settings.json', 'r', encoding='utf-8') as jf:
                                settings = json.load(jf)
                            if settings["scripts"] == []:
                                print("[!] Скрипты не добавлены. Добавте их командой /add_script")
                            for script in settings["scripts"]:
                                print(f" - {script}")
                                
                        elif command == "/add_script":
                            add_script = input("Введите название файла (без расширения) (/ex для отмены): ")
                            if add_script == "/ex":
                                print("[/] Отменено")
                            else:
                                with open('settings.json', 'r', encoding='utf-8') as jf:
                                    settings = json.load(jf)
                                script_in_file = False
                                for script in settings["scripts"]:
                                    if script == add_script+".py":
                                        print("[!] Данный скрипт уже добавлен. Загрузите его командой /load_script")
                                        script_in_file = True
                                        break
                                    script_in_file = False
                                if script_in_file is False:
                                    if os.path.exists(f"loads/{add_script}.py"):
                                        settings["scripts"].append(add_script+".py")
                                        with open('settings.json','w', encoding='utf-8') as jf:
                                            json.dump(settings, jf, indent = 4)
                                        print("[/] Скрипт добавлен")
                                    else:
                                        print("[!] Данный файл не найден в директории loads")
                                    
                        elif command == "/del_script":
                            del_script = input("Введите название файла (без расширения) (/ex для отмены): ")
                            if del_script == "/ex":
                                print("[/] Отменено")
                            else:
                                with open('settings.json', 'r', encoding='utf-8') as jf:
                                    settings = json.load(jf)
                                script_in_file = False
                                for script in settings["scripts"]:
                                    if script == del_script+".py":
                                        script_in_file = True
                                        break
                                    script_in_file = False
                                if script_in_file:
                                    settings["scripts"].remove(del_script+".py")
                                    with open('settings.json','w', encoding='utf-8') as jf:
                                        json.dump(settings, jf, indent = 4)
                                    print("[/] Скрипт удалён")
                                else:
                                    print("[!] Данный скрипт не добавлен. Закиньте файл скрипта в директорию программы в папку \"loads\" и добавте его командой /add_script")
                                    
                        elif command == "/load_script":
                            with open('settings.json', 'r', encoding='utf-8') as jf:
                                settings = json.load(jf)
                            load_script = input("Укажите название скрипта, который хотите загруить (без расширения) (/ex для отмены): ")
                            if load_script == "/ex":
                                print("[/] Отменено")
                            else:
                                if os.path.exists(f"loads/{load_script}.py"):
                                    if load_script+".py" in settings["scripts"]:
                                        os.system('python loads/{}.py'.format(load_script))
                                    else:
                                        print("[!] Данный скрипт не добавлен. Закиньте файл скрипта в директорию программы в папку \"loads\" и добавте его командой /add_script")
                                else:
                                    print("[!] Данный файл не существует в директории loads")
                            
                        elif command == "/create_script":
                            with open('settings.json', 'r', encoding='utf-8') as jf:
                                settings = json.load(jf)
                            create_script = input("Введите название скрипта без расширения (/ex для отмены): ")
                            if create_script == "/ex":
                                print("[/] Отменено")
                            elif os.path.exists(f"loads/{create_script}.py"):
                                print("[!] Данный скрипт уже существует в директории loads")
                            else:
                                file = os.open(f"loads/{create_script}.py", os.O_CREAT|os.O_RDWR)
                                print("[/] Скрипт успешно создан! Редактируйте его командой /rescript")
                                
                        elif command == "/phone":
                            from phonenumbers import carrier, geocoder, timezone
                            number = input("Введите номер телефона: ")
                            phoneNumber = phonenumbers.parse(number) 
                            carrier = carrier.name_for_number(phoneNumber, 'ru')
                            reg = geocoder.description_for_number(phoneNumber, 'ru') 
                            timeZone = timezone.time_zones_for_number(phoneNumber)
                            valid = phonenumbers.is_valid_number(phoneNumber)
                            if valid:
                                print("=============== ИНФОРМАЦИЯ О НОМЕРЕ ===============")
                                print(f" Существует: {valid}")
                                print(f" Оператор: {carrier}")
                                print(f" Зарегистрирован: {reg}")
                                print(" Часовые пояса:")
                                for tz in timeZone:
                                    print(f"  - {tz}")
                                print("===================================================")
                            else:
                                print("[!] Данного номера не существует")
                        
                        elif command == "/calc":
                            def calculate(expression):
                                try:
                                    result = eval(expression)
                                    return result
                                except Exception as e:
                                    return "[!] Ошибка при вычислении"
                            expression = input("Введите выражение для вычисления (/ex для отмены): ")
                            if expression.lower() == "/ex":
                                print("[/] Отменено")
                                break
                            result = calculate(expression)
                            print("Результат: {}".format(result))
                        
                        else:
                            print("[!] Неизвестная комманда. Список команд: /help")
                    else:
                        with open('data.json', 'r', encoding='utf-8') as jf:
                            fd = json.load(jf)
                        # if messages_cooldown_bool is False:
                            # messages_cooldown_bool = not messages_cooldown_bool
                        print(f"|Чат| {fd['user'][id-1]['privileg']} {nick} ({id}) написал -> {command}")
                            # messages_cooldown_bool = False

            except Exception as e:
                print("\nВозникла внутренняя ошибка")
                with open('data.json', 'r', encoding='utf-8') as jf:
                    fd = json.load(jf)
                if fd['user'][id-1]['privileg'] == privileg[4] or fd['user'][id-1]['privileg'] == privileg[3]:
                    print(f"{e}\n")
        except KeyboardInterrupt:
            pass
                
# Проверка на изменение цвета текста
try:
    def check_color():
        with open('settings.json', 'r', encoding='utf-8') as jf:
            settings = json.load(jf)
        if settings['color'] == "GREEN":
            print(Fore.GREEN)
        elif settings['color'] == "RED":
            print(Fore.RED)
        elif settings['color'] == "BLUE":
            print(Fore.BLUE)
        elif settings['color'] == "YELLOW":
            print(Fore.YELLOW)
        elif settings['color'] == "PURPLE":
            print(Fore.MAGENTA)
        elif settings['color'] == "WHITE":
            print(Fore.WHITE)
        elif settings['color'] == "BLACK":
            print(Fore.BLACK)
        else:
            print(Fore.WHITE)
except Exception as e:
    print("\nВозникла внутренняя ошибка")
    
# Проверка на изменение фона текста
try:
    def check_bgcolor():
        with open('settings.json', 'r', encoding='utf-8') as jf:
            settings = json.load(jf)
        if settings['bgcolor'] == "GREEN":
            print(Back.GREEN)
        elif settings['bgcolor'] == "RED":
            print(Back.RED)
        elif settings['bgcolor'] == "BLUE":
            print(Back.BLUE)
        elif settings['bgcolor'] == "YELLOW":
            print(Back.YELLOW)
        elif settings['bgcolor'] == "PURPLE":
            print(Back.MAGENTA)
        elif settings['bgcolor'] == "WHITE":
            print(Back.WHITE)
        elif settings['bgcolor'] == "BLACK":
            print(Back.BLACK)
        else:
            print(Back.BLACK)
except Exception as e:
    print("\nВозникла внутренняя ошибка")

# Вход в аккаунт
try:
    attempt = 3
    def login():
        global attempt
        with open('settings.json', 'r+', encoding='utf-8') as jf:
            settings = json.load(jf)
        if settings['session']['name'] == "" and settings['session']['password'] == "" or settings['device_name'] != device_name:
            def register():
                user_in_file = False
                print("Зарегистрируйтесь. Чтобы авторизироваться введите /login")
                nick = input("[РЕГИСТРАЦИЯ] Создайте свой никнейм: ")
                if nick == "/login":
                    login()
                else:
                    password = input("[РЕГИСТРАЦИЯ] Создайте пароль: ")
                    if nick == "" or password == "" or " " in nick or " " in password or "/" in nick or "{" in nick or "}" in nick or nick == "No-Name":
                        print("[!] Никнейм или пароль содержит запрещённые символы")
                        register()
                    else:
                        if len(nick) < 3 or len(nick) > 15:
                            print("[!] Никнейм не должен быть менее 16 символов и более 2 символов")
                            register()
                        else:
                            with open('data.json', 'r+', encoding='utf-8') as json_file:
                                data = json.load(json_file)
                            for user in data['user']:
                                if nick == user['name']:
                                    print("[!] Данный никнейм уже используется")
                                    user_in_file = True
                            if user_in_file is False:
                                temp_id = 1
                                for i in data["id"]:
                                    temp_id += 1
                                a_dict_userid = {
                                    nick: {
                                        "id": temp_id,
                                        "name": nick
                                    }
                                }
                                a_dict_user = {
                                    "name": nick,
                                    "password": password,
                                    "id": temp_id,
                                    "friends": [],
                                    "friends_request": [],
                                    "friends_your_request": [],
                                    "ban": False,
                                    "reason": "нет причины",
                                    "who_ban": "No-Name",
                                    "balance": 10000,
                                    "use_promocode": False,
                                    "privileg": "Пользователь",
                                    "reputation": 0,
                                    "comm": 0,
                                    "list": "",
                                    "title": "Я пользователь данной программы",
                                    "404": False
                                }
                                with open('data.json','r+', encoding='utf-8') as file:
                                    file_data = json.load(file)
                                    file_data["id"].append(a_dict_userid)
                                    file_data["user"].append(a_dict_user)
                                    file.seek(0)
                                    json.dump(file_data, file, indent = 4)
                                file_data = None
                                login()
                            else:
                                register()
            if attempt > 0:
                print("Авторизируйтесь. Чтобы зарегистрировться введите /reg")
                nick = input("[АВТОРИЗАЦИЯ] Введите свой ник: ")
                if nick == "/reg":
                    register()
                else:
                    password = input("[АВТОРИЗАЦИЯ] Введите пароль: ")
                    check_user_profile = False

                    if nick == "" or password == "" or " " in nick or " " in password or "/" in nick or "{" in nick or "}" in nick or nick == "No-Name":
                        print("[!] Никнейм или пароль содержит запрещённые символы")
                        login()
                    else:
                        if len(nick) < 3 or len(nick) > 15:
                            print("[!] Никнейм не должен быть менее 16 символов и более 2 символов")
                            login()
                        else:
                            with open('data.json', 'r+', encoding='utf-8') as json_file:
                                data = json.load(json_file)
                            for user in data['user']:
                                check_user = user['name']
                                if nick == check_user:
                                    check_password = user['password']
                                    if password == check_password:
                                        check_user_profile = True
                                    else:
                                        print("[!] Неверный пароль")
                                        attempt -= 1
                                        login()
                            if check_user_profile is False:
                                print("[!] Такого пользователя не существует. Введите /reg для создания аккаунта")
                                login()
                            else:
                                with open('settings.json', 'r', encoding='utf-8') as jf:
                                    settings = json.load(jf)
                                settings['session']['name'] = nick
                                settings['session']['password'] = password
                                settings['device_name'] = device_name
                                with open('settings.json', 'w') as jf:
                                    json.dump(settings, jf, indent=4)
                                data = None
                                os.system("cls")
                                check_color()
                                check_bgcolor()
                                preview_text = Figlet(font='slant')
                                print(preview_text.renderText('MANAGER'))
                                print(f"=============================================================================================\n Добро пожаловать, {nick}, в консольную программу. Список команд ---> /help \n=============================================================================================")
                                programm(nick)
            else:
                print("[!] Попытки авторизироваться закончились! Попробуйте войти позже")
                time.sleep(2)
                sys.exit()
        else:
            os.system("cls")
            check_color()
            check_bgcolor()
            preview_text = Figlet(font='slant')
            print(preview_text.renderText('MANAGER'))
            nick = settings['session']['name']
            print(f"=============================================================================================\n Добро пожаловать, {nick}, в консольную программу. Список команд ---> /help \n=============================================================================================")
            programm(nick)
    login()
except Exception as e:
    print("\nВозникла внутренняя ошибка\n")