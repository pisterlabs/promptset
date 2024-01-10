# /usr/bin/python3
# -*- coding: utf-8 -*-
# V1.4 database prefs replaced by settings.ini file
# Thanks to Oceane, JS, Montaulab, Claudine, CQuest, Musée du minitel....
# And the Wishwizardteam Olli, Holger, Mephy und allen anderen.
# *************************************************************

import csv, sys, pynitel
import time, configparser, random
import pyhid_usb_relay, requests
from openai import OpenAI
from threading import Thread
from escpos.printer import *
from kerykeion import AstrologicalSubject  # ,  Report


####
# Utility functions
####
# ##****** replace short sign to long
def get_data_astro(dat_ast, lan):
    global kanye, language
    kanye = None
    try:
        kanye = AstrologicalSubject(*dat_ast)
    except:
        m.home()
        m.message(20, 5, 3, "There is an issue with your data, please try again")
        # self.changeState(self.stateWelcome)
    try:
        print("In try: ", lan)
        language = get_language_by_country_code(lan)
        print("language")
    except:
        m.home()
        m.message(20, 5, 3, "There is an issue with your language data, please try again")


def get_language_by_country_code(country_code):
    print("Given CCode: " + country_code)
    try:
        url = f"https://restcountries.com/v2/alpha/{str(country_code)}"
        response = requests.get(url)

        if response.status_code == 200:
            data_count = response.json()
            languages = data_count.get("languages", [])
            if languages:
                return languages[0].get("name", "Unbekannte Sprache")
            else:
                return "English"
        else:
            return "English"
    except Exception as e:
        return "English"


def signs(sign):
    x = ""
    signdata = [["Can", "Cancer"], ["Leo", "Lion"], ["Vir", "Virgo"], ["Lib", "Libra"],
                ["Sco", "Scorpio"], ["Sag", "Sagittarius"], ["Cap", "Capricorn"],
                ["Aqu", "Aquarius"], ["Pis", "Pisces"], ["Ari", "Aries"],
                ["Tau", "Taurus"], ["Gem", "Gemini"]
                ]

    search1 = sign
    for i, sign in enumerate(signdata):
        if search1 in sign[0]:
            x = sign[1]
            print(x)
    return x


# ****** prepare USB relays
def usbRelaysCheck():
    global relay
    relay = None
    try:
        relay = pyhid_usb_relay.find()
        print("USB RELAY connected")
        return True
    except:
        print("No USB RELAY found")
        relay = None
        return False


# ****** Write csv datafile to dataobject "data"
def create_data(csv_dateipfad):
    data = {}
    # CSV-Datei öffnen und auslesen
    with open(csv_dateipfad, "r") as csv_datei:
        csv_reader = csv.reader(csv_datei, delimiter=";")
        spaltenueberschriften = next(csv_reader)
        for zeile in csv_reader:
            beschreibung = zeile[0]
            spalten_werte = {}
            for i in range(1, len(zeile)):
                spaltenueberschrift = spaltenueberschriften[i]
                wert = zeile[i]
                spalten_werte[spaltenueberschrift] = wert
            data[beschreibung] = spalten_werte
    return data


# ************Find actual IP in network
def getNetworkIp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.connect(('<broadcast>', 0))
    return s.getsockname()[0]


def getMilliseconds():
    return int(round(time.time() * 1000))


def transl(beschreibung):
    global lang, data
    # Überprüfen, ob die Beschreibung im Datenobjekt vorhanden ist
    if beschreibung in data:
        # Überprüfen, ob die Spaltenüberschrift im Datenobjekt vorhanden ist
        if lang in data[beschreibung]:
            # Wert aus dem Datenobjekt abrufen und zurückgeben
            wert = data[beschreibung][lang]
            return wert
        else:
            print("Spaltenüberschrift nicht gefunden: ")
            return "problem"
    else:
        print("Beschreibung nicht gefunden: ", beschreibung)


def randomFileFromFolder(dossier):  # ****Random audiofile from directory
    lst = os.listdir(dossier)
    f = random.randint(0, len(lst) - 1)
    path = dossier + "%s" % lst[f]  # .split(".")[0]
    print(path)
    return path


# ******Format to left and to right
def strformat(left='', right='', fill=' ', width=40):
    # " formattage de texte "
    total = width - len(left + right)
    if total > 0:
        out = left + fill * total + right
    else:
        out = left + right
    return out


# ******Screenprint centered
def strcenter(row=0, pos=20, txt=' ', width=40, size=0):
    out = pos - len(txt) // 2
    if out < 1:
        out = 1
    m.pos(row, out)
    m.scale(size)
    m._print(txt)
    m.scale(1)
    return ()


def printCheck():
    global p, pV, sltime
    config = configparser.ConfigParser()
    config.read('settings.ini')
    p = None
    # check the time before the screen automaticly changes ( Screensaver)
    sltime = int(config['prefs']['timer'])

    # lang = config['prefs']['lang_code']  *****************************************???????????????? oder anders????

    # Check and prepare Printer (First USB, then serial) (Serial not finished)
    try:
        p = Usb(0x04b8, 0x0e1f)
        # p = Usb(int(config['printer']['p_idvend'],16), int(config['printer']['p_idprod'],16))#, r1[6], r1[7])
        # p = Usb(int(r1[4],16), int(r1[5],16))#, r1[6], r1[7])
        pV = True
        p.charcode = 'CP850'
        p.codepage = 'CP850'
        # p.profile.media.width.pixel = 380
        print("USBPrinter ? ", pV)
        # return True
    except:
        pV = False
        print("USBPrinter ? ", pV)
        # return False

    if not pV:
        try:
            p = Serial(config['printer']['p_idvend'], 9600, 8, 1)  # to complete
            pV = True
            print("Serial Printer ? ", pV)
        except:
            pV = False
            print("SerialPrinter ? ", pV)
    return pV


# Function to send a message to the OpenAI chatbot model and return its response
def send_message(message_log):
    global answer, client
    config = configparser.ConfigParser()
    config.read('settings.ini')
    # Use OpenAI's ChatCompletion API to get the chatbot's response
    response = client.chat.completions.create(
        model=config['IA']['ia_model'],
        messages=message_log,
        max_tokens=int(config['IA']['max_tok']),
        temperature=1.0
    )

    token_usage = response.usage.total_tokens
    print("Anzahl der verwendeten Tokens:", token_usage)
    # Find the first response from the chatbot that has text in it (some responses may not have text)
    # for choice in response.choices:
    # if "text" in choice:
    # return choice.text
    # print(response) #.choices[0].message.content)

    # If no response with text is found, return the first response's content (which may be empty)
    return response.choices[0].message.content


def chatbot(messa):
    global answer, client
    config = configparser.ConfigParser()
    config.read('settings.ini')
    # Initialize the conversation history with a message from the chatbot
    txt = ("Make a prediction for the year 2024. Use the given data. " +
           "Use familiar, magical and mystical language full of fantasy like an old fortuneteller on a fairy fair. " +
           "Use the astrologic and numerologic  informations from the personal data. " +
           "Use also the extra text as special question if given. " +
           "Integrate in the beginning the sign, ascendent and the most important data from the numerological results. " +
           "If a different language is asked for, please translate. " +
           "Give a short answer witch doesn't use more than approximately " + config['IA']['max_tok'] + " tokens. " +
           "Always finish the sentences.")
    # print(txt)
    message_log = [{"role": "system", "content": txt}]

    # Set a flag to keep track of whether this is the first request in the conversation
    first_request = True

    # Start a loop that runs until the user types "quit"
    while True:
        if first_request:
            # If this is the first request, get the user's input and add it to the conversation history
            # user_input = input("You: ")
            user_input = messa
            message_log.append({"role": "user", "content": user_input})
            # Send the conversation history to the chatbot and get its response
            response = send_message(message_log)
            # Add the chatbot's response to the conversation history and print it to the console

            answer = response
            # Set the flag to False so that this branch is not executed again
            first_request = False
        else:
            # If this is not the first request, get the user's input and add it to the conversation history
            user_input = input("You: ")
            user_input = user_input
            # If the user types "quit", end the loop and print a goodbye message
            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            message_log.append({"role": "user", "content": user_input})

            # Send the conversation history to the chatbot and get its response
            response = send_message(message_log)

            # Add the chatbot's response to the conversation history and print it to the console
            message_log.append({"role": "assistant", "content": response})
        answer = response
        return


def split_string_into_lines(text, max_line_length=40):
    words = text.split()  # Teile den Text in Worte auf
    lines = []
    current_line = ""

    for word in words:
        # Wenn das Hinzufügen des aktuellen Wortes zur aktuellen Zeile die maximale Länge
        # überschreiten würde, füge die Zeile zu den Zeilen hinzu und starte eine neue Zeile.
        if len(current_line) + len(word) + 1 <= max_line_length:
            if current_line:
                current_line += " "
            current_line += word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


###
# State machine aka "The brain"
###
class StateMachine:
    def __init__(self):
        self.stack = []
        self.push(self.stateInit)
        self.radarCount = 0
        self.time = 0
        self.time1 = 0
        self.halftime = 0
        self.offStageLimit = 0
        self.offStageMessage = 0
        self._mess = ""

    def mess_change(self):
        return self._mess

    def push(self, state):
        self.stack.append(state)
        self.changing = True

    def pop(self):
        self.stack.pop()

    def changeState(self, state):
        self.pop()
        self.push(state)

    def update(self):
        if (len(self.stack) <= 0):
            return

        entering = self.changing
        self.changing = False
        self.stack[-1](entering)

    # *****************************************************

    def stateInit(self, entering):

        print("~~~ Initialisation ~~~")
        print("Minitel init")
        global m, lang, th1, data, sltime, p, pV, answer, client

        print("Officielle IP: ", getNetworkIp())
        config = configparser.ConfigParser()
        config.read('settings.ini')
        sltime = config['prefs']['timer']
        lang = config['prefs']['lang_code']
        speed = int(config['prefs']['speed'])
        print("Language: ", lang, " Screensaver/s: ", sltime)
        m = pynitel.Pynitel(serial.Serial('/dev/ttyUSB0', 1200, parity=serial.PARITY_EVEN, bytesize=7, timeout=2))
        # os.system("echo -en '\x1b\x3a\x6b\x64' > /dev/ttyUSB0")
        os.system("stty -F /dev/ttyUSB0 speed 1200")
        if speed == 4800:
            os.system("echo -en '\x1b\x3a\x6b\x76' > /dev/ttyUSB0")
            m.end()
            os.system("stty -F /dev/ttyUSB0 speed 4800")
            m = pynitel.Pynitel(serial.Serial('/dev/ttyUSB0', 4800, parity=serial.PARITY_EVEN, bytesize=7, timeout=2))
            print("Baudrate: ", speed)
        else:  # speed == 1200:
            print("Baudrate: ", speed)

        # ###*******Prepare usb relais
        r = usbRelaysCheck()
        if r:
            print("Relais to true")
            relay.set_state(1, True)
            relay.set_state(4, True)
            time.sleep(2)
            print("Relais to False")
            relay.set_state(1, False)
            relay.set_state(4, False)

        # ###*******Prepare prefs :sltime, printer
        printCheck()

        # ###***** Prepa ChatGPT API
        os.system("echo -en '\x1b\x3a\x6A\x43' > /dev/ttyUSB0")

        api_key = config['IA']['api_key']
        client = OpenAI(api_key=config['IA']['api_key'])

        if not api_key:
            m.message(10, 5, 4, "No API KEY found, enter in settings.ini or on next page", True)

            # print("was is hier los")
            time.sleep(5)
            self.changeState(self.statePrefs(""))
        else:
            print("else")

        answer = ""
        print("~~~ Initialisation Done ~~~", "\n")
        self.changeState(self.stateFortunePic)

    def stateFortunePic(self, entering):
        global lang, sltime, ver

        print("Fortune Teller picture")
        rd = random.randint(1, 5)
        if rd == 1:
            r = usbRelaysCheck()
            if r:
                relay.set_state(1, True)
                relay.set_state(4, True)
                time.sleep(1)
                relay.set_state(4, False)
                time.sleep(4)
                relay.set_state(1, False)
        m.home()

        m.load(1, './WM/fortune.vdt')
        m.draw(1)
        while True:
            # ligne finale
            m.pos(24, 1)
            m.color(m.vert)
            m._print(transl("p0t1"))
            m.pos(24, 19)
            m.inverse()
            m.color(m.cyan)
            m._print('  ENVOI ')
            # attente saisie
            m.cursor(False)
            (choix1, touche) = m.input(23, 1, 1, sltime, "")

            if touche != "":
                break
            elif choix1 != "":
                break

        if touche != "" or choix1 != "":
            self.changeState(self.stateWelcome)

    def stateLanguage(self, entering):  # p0t1
        print("State language")
        global lang, th1, sltime, ver, choix1
        choix1 = ""
        touche = ""
        m._del(0, 0)
        while True:  # t2 > time.time():#True:
            if True == True:  # affichage
                # entête sur 2 lignes + séparation
                m.home()
                m.pos(0)
                m.pos(0, 0)
                m._print("V" + ver + " / " + lang)
                m.pos(2)
                m.pos(2, 1)
                m.scale(3)
                m._print(transl("p1t1"))
                m.pos(4, 1)
                m.scale(3)
                m._print(transl("p1t2"))
                m.scale(0)
                strcenter(row=3, pos=27, txt=transl("p1t3"), width=40, size=0)
                strcenter(row=4, pos=27, txt=transl("p1t3a"), width=40, size=0)
                m.pos(5)
                m.color(m.bleu)
                m.plot('̶', 40)
                strcenter(row=8, pos=20, txt=transl("p1t4"), width=40, size=0)
                # strcenter(row = 9,pos=20, txt = 'choice', width = 40, size = 0)
                strcenter(row=11, pos=10, txt=transl("p1t5"), width=40, size=0)
                strcenter(row=11, pos=30, txt=transl("p1t6"), width=40, size=0)
                strcenter(row=13, pos=10, txt=transl("p1t7"), width=40, size=0)
                strcenter(row=13, pos=30, txt=transl("p1t8"), width=40, size=0)
                # ligne finale
                m.pos(21)
                m.color(m.bleu)
                m.plot('̶', 40)
                m.pos(22, 1)

                m.pos(24, 1)
                m.color(m.vert)
                m._print(transl("p1t9"))
                m.pos(24, 8)
                m.inverse()
                m.color(m.cyan)
                m._print("GUIDE")

                m.pos(24, 22)
                m.color(m.vert)
                m._print(transl("p1t10"))
                m.pos(24, 36)
                m.inverse()
                m.color(m.cyan)
                m._print("ENVOI")
            else:
                break
                # page = abs(page)

            (choix1, touche) = m.input(24, 33, 2, sltime)
            m.cursor(False)

            if touche == m.suite:
                return (touche)

            elif touche == m.chariot:
                break

            elif touche == m.retour or touche == m.annulation:
                break
            elif touche == m.envoi:
                break
            elif touche == m.sommaire:
                # print("touche sommaire")
                break

            elif touche == m.guide:
                break
            elif touche == m.correction:  # retour saisie pour correction
                return (touche)

        if choix1 == "":
            choix1 = lang
        if touche == m.sommaire and choix1 == "sleep":
            # choix1 = ""
            print("Go to sleeper")
            self.changeState(self.stateWelcome)
        if touche == m.envoi and choix1 == "FR":
            lang = "FR"
            m.message(16, 7, 2, transl("p1t11"), bip=False)
            self.changeState(self.stateWelcome)
        elif touche == m.envoi and choix1 == "EN":
            lang = "EN"
            m.message(16, 7, 2, transl("p1t11"), bip=False)
            self.changeState(self.stateWelcome)
        elif touche == m.envoi and choix1 == "DE":
            lang = "DE"
            m.message(16, 7, 2, transl("p1t11"), bip=False)
            self.changeState(self.stateWelcome)
        elif touche == m.envoi and choix1 == "ES":
            lang = "ES"
            m.message(16, 7, 2, transl("p1t11"), bip=False)
            self.changeState(self.stateWelcome)

        elif touche == m.envoi:
            if choix1 != "FR" or choix1 != "EN" or choix1 != "DE" or choix1 != "ES":
                m.home()
                m.message(15, 7, 3, "Wrong CODE, try again ", bip=True)
        elif touche == m.guide:
            self.changeState(self.stateInfo1)  # TODO HELPPAGE

    def stateInfo1(self, entering):
        print("Infopage")
        time.sleep(2)
        self.changeState(self.stateWelcome)

    def statePrefs(self, entering):  # Prefs: timer screensave, perhaps, Printer
        print("State Preferences")
        global lang, sltime

        m.resetzones()
        config = configparser.ConfigParser()
        config.read('settings.ini')

        teil1 = config['IA']['api_key'][:27]
        teil2 = config['IA']['api_key'][27:]
        # print(config['IA']['api_key'])
        # print(teil1,teil2)
        m.zone(7, 15, 23, config['prefs']['ip_adr'], m.blanc)
        m.zone(7, 36, 23, config['prefs']['speed'], m.blanc)
        m.zone(9, 15, 23, config['prefs']['timer'], m.blanc)
        m.zone(9, 36, 23, config['prefs']['lang_code'], m.blanc)
        m.zone(11, 15, 23, config['printer']['p_idvend'], m.blanc)
        m.zone(12, 15, 23, config['printer']['p_idprod'], m.blanc)
        m.zone(13, 15, 23, config['printer']['p_timer'], m.blanc)
        m.zone(14, 15, 23, config['printer']['p_3'], m.blanc)
        m.zone(15, 15, 23, config['printer']['p_4'], m.blanc)
        m.zone(17, 10, 23, teil1, m.blanc)
        m.zone(18, 10, 23, teil2, m.blanc)
        m.zone(19, 10, 23, config['IA']['ia_model'], m.blanc)
        m.zone(19, 36, 23, config['IA']['max_tok'], m.blanc)
        m.zone(20, 15, 23, "NO", m.blanc)
        m.zone(20, 35, 23, "NO", m.blanc)
        # r = res[0]
        touche = m.repetition
        zone = 1
        m.home()

        while True:
            m.home()
            m.pos(0, 1)
            m._print("Version " + ver)
            m.pos(2, 1)
            m.scale(3)
            m._print(transl("p3t1"))
            m.pos(4, 1)
            m.scale(3)
            m._print(transl("p3t2"))
            m.scale(0)
            strcenter(row=1, pos=28, txt=transl("p3t3"), width=40, size=0)
            strcenter(row=3, pos=28, txt=transl("p3t4"), width=40, size=0)
            strcenter(row=4, pos=28, txt=transl("p3t5"), width=40, size=0)
            m.pos(5)
            m.color(m.bleu)
            m.plot('̶', 40)
            m.pos(7)
            m._print('' + strformat(left="Local / IP"[:11], right="*                        ", width=39))
            strcenter(row=7, pos=30, txt="Bauds ", width=20, size=0)
            m.pos(9)
            m._print('' + strformat(left="Time sleep"[:11], right="*                        ", width=39))
            strcenter(row=9, pos=30, txt="Language ", width=20, size=0)
            m.pos(11)
            m._print('' + strformat(left="Print Val1"[:11], right="-                        ", width=39))
            m.pos(12)
            m._print('' + strformat(left="Print Val2"[:11], right="-                        ", width=39))
            m.pos(13)
            m._print('' + strformat(left="Print Val3"[:11], right="-                        ", width=39))
            m.pos(14)
            m._print('' + strformat(left="Print Val4"[:11], right="-                        ", width=39))
            m.pos(15)
            m._print('' + strformat(left="Print Val5"[:11], right="-                        ", width=39))
            m.pos(17)
            m._print('' + strformat(left="IA API"[:11], right="*..........................", width=39))
            m.pos(18)
            m._print('' + strformat(left=""[:1], right="*..........................", width=39))
            m.pos(19)
            m._print('' + strformat(left="Model"[:11], right="*                        ", width=39))
            strcenter(row=19, pos=30, txt="Max Token: ", width=20, size=0)
            m.pos(20)
            m._print('' + strformat(left="Reboot"[:11], right="*........................", width=39))
            strcenter(row=20, pos=30, txt="Shutdown ", width=40, size=0)
            # ligne finale
            m.pos(21)
            m.color(m.bleu)
            m.plot('̶', 40)
            m.pos(22, 1)
            m._print(transl("p3t11"))
            m.pos(23, 1)

            m.color(m.vert)
            m._print(transl("p3t12"))
            m.pos(23, 12)
            m.inverse()
            m.color(m.cyan)
            m.underline()
            m._print("_SUITE ")

            m.pos(24, 1)
            m.color(m.vert)
            m._print(transl("p3t13"))
            m.pos(24, 12)
            m.inverse()
            m.color(m.cyan)
            m._print(" RETOUR")
            m.pos(24, 25)
            m._print(transl("p3t15"))
            m.pos(24, 33)
            m.inverse()
            m.color(m.cyan)
            m._print("SOMMAIRE")
            m.pos(22, 21)
            m._print(transl("p3t14"))
            m.pos(22, 33)
            m.inverse()
            m.color(m.cyan)
            m._print('  ENVOI ')
            m.pos(24, 28)

            # gestion de la zone de saisie courante
            (zone, touche) = m.waitzones(zone, sltime)
            # print(touche)

            if touche == 1:
                print("Touche ENVOI: " + str(touche))
                choice = ""
                if m.zones[0] == "":
                    m.zones[0] = "localhost"
                api = (m.zones[9]['texte'].strip() + m.zones[10]['texte'].strip()).strip()
                config['prefs'] = {
                    'IP_adr': m.zones[0]['texte'],
                    'speed': m.zones[1]['texte'],
                    'timer': m.zones[2]['texte'],
                    'lang_code': m.zones[3]['texte']}
                config['printer'] = {
                    'p_idvend': m.zones[4]['texte'],
                    'p_idprod': m.zones[5]['texte'],
                    'p_timer': m.zones[6]['texte'],
                    'p_3': m.zones[7]['texte'],
                    'p_4': m.zones[8]['texte']}
                config['IA'] = {
                    'api_key': api,
                    'ia_model': m.zones[11]['texte'],
                    'max_tok': m.zones[12]['texte'],
                }
                with open('settings.ini', 'w') as configfile:
                    config.write(configfile)

                sltime = int(m.zones[2]['texte'])
                lang = m.zones[3]['texte']

                break
            if touche == 3:
                break
            if touche == 5:
                break
            if touche == 6:
                break
        if touche == 1:
            print("check & prepare data ")
            m.resetzones()
            self.changeState(self.stateWelcome)
        elif touche == 3 and str(m.zones[5]['texte']) == "YES":
            m.home()
            m.message(15, 7, 3, "Go to reboot ", bip=True)
            m.resetzones()
            os.system('shutdown now -h')


        elif touche == 3 and m.zones[6]['texte'] == "YES":
            m.home()
            m.message(15, 7, 3, "Go to shutdown", bip=True)
            m.resetzones()
            os.system('shutdown now -h')
        elif touche == 6:
            m.resetzones()
            self.changeState(self.stateWelcome)
        else:
            # print(m.zones[2], " ", m.zones[3])
            print("stay in stream")

    def stateWelcome(self, entering):
        print("State Welcome")
        global lang, sltime, data, answer
        answer = ""
        touche = 0
        choix1 = ""
        m.home()
        while True:
            if True == True:  # affichage
                # entête sur 2 lignes + séparation
                m.home()
                m.pos(2)
                m.pos(2, 1)
                m.scale(3)
                m._print(transl("p2t1"))
                m.pos(4, 1)
                m.scale(3)
                m._print(transl("p2t2"))
                m.scale(0)
                strcenter(row=3, pos=28, txt=transl("p2t3"), width=40, size=0)
                strcenter(row=4, pos=28, txt=transl("p2t3a"), width=40, size=0)
                m.pos(5)
                m.color(m.bleu)
                m.plot('̶', 40)
                strcenter(row=8, pos=20, txt=transl("p2t4"), width=40, size=0)
                strcenter(row=9, pos=20, txt=transl("p2t5"), width=40, size=0)
                strcenter(row=10, pos=20, txt=transl("p2t6"), width=40, size=0)
                strcenter(row=13, pos=20, txt=transl("p2t7"), width=40, size=3)
                # strcenter(row = 13,pos=20, txt = transl("p2t8"), width = 40, size = 3)
                # strcenter(row = 8,pos=30, txt = transl("p2t9"), width = 40, size = 0)
                # strcenter(row = 9,pos=30, txt = transl("p2t10"), width = 40, size = 0)
                # strcenter(row = 10,pos=30, txt = transl("p2t11"), width = 40, size = 0)
                # strcenter(row = 13,pos=30, txt = transl("p2t12"), width = 40, size = 3)
                strcenter(row=16, pos=20, txt=transl("p2t13"), width=40, size=3)
                strcenter(row=18, pos=20, txt=transl("p2t14"), width=40, size=0)

                # ligne finale
                m.pos(21)
                m.color(m.bleu)
                m.plot('̶', 40)
                m.pos(22, 1)

                m.color(m.vert)
                m.pos(24, 1)
                m._print(transl("p2t15"))
                m.pos(24, 10)
                m.inverse()
                m.color(m.cyan)
                m._print("SOMMAIRE")
                m.pos(23, 1)
                m._print(transl("p2t16"))
                m.pos(23, 10)
                m.inverse()
                m.color(m.cyan)
                m._print("GUIDE")
                m.pos(24, 24)
                m.color(m.vert)
                m._print(transl("p2t17"))
                m.pos(24, 36)
                m.inverse()
                m.color(m.cyan)
                m._print("ENVOI")

            else:
                break
                # page = abs(page)

            (choix1, touche) = m.input(24, 33, 2, sltime)
            # ****** check for integer
            if choix1 != "sleep":
                if not isinstance(choix1, int):
                    if not choix1.isdigit():
                        choix1 = 0
                    choix1 = int(choix1)

            m.cursor(False)
            if touche == m.suite:
                return touche
            elif touche == m.chariot:
                break
            elif touche == m.retour or touche == m.annulation:
                break
            elif touche == m.envoi:
                break
            elif touche == m.sommaire:
                break
            elif touche == m.guide:
                break
            elif touche == m.correction:  # retour saisie pour correction
                return touche
            elif touche != m.repetition:
                m.bip()
        # print("Bin hier")
        if touche == m.envoi and choix1 == 1:
            m.home()
            self.changeState(self.stateEnterData1)

        elif touche == m.envoi and choix1 == 98:

            self.changeState(self.statePrefs)
        elif touche == m.envoi:
            if choix1 < 1 or choix1 > 2 or choix1 == 99:
                m.resetzones()
                m.message(15, 7, 1.5, "Wrong Number, try again ", bip=True)
        elif touche == m.sommaire and choix1 == "sleep":
            print("sommaire to sleep")
            self.changeState(self.stateFortunePic)  # ***new ?????
            # return
        elif touche == m.sommaire and choix1 == 0:
            print("sommaire to Language")
            m.resetzones()
            self.changeState(self.stateLanguage)
        if touche == m.guide:
            self.changeState(self.stateInfo1)

    # ****************************************  ASTRO DATA
    def stateEnterData1(self, entering):  # Enter Name and Sexe
        global data, lang, sltime, data_p, data_gender
        # data_p = ()
        annu_save = ""
        print("State Enter Name / Sexe")
        zone = 1
        m.home()
        # m.resetzones()
        m.zone(10, 8, 23, '', m.blanc)
        m.zone(17, 20, 1, '', m.blanc)
        while True:

            # HEADLINE ******************
            m.pos(2)
            m.pos(2, 1)
            m.scale(3)
            m._print(transl("p10t1"))
            m.pos(4, 1)
            m.scale(3)
            m._print(transl("p10t2"))
            m.scale(0)
            # strcenter(row = 1,pos=28, txt = transl("p10t3"), width = 40, size = 0)
            strcenter(row=3, pos=28, txt=transl("p10t3"), width=40, size=0)
            strcenter(row=4, pos=28, txt=transl("p10t3a"), width=40, size=0)
            m.pos(5)
            m.color(m.bleu)
            m.plot('̶', 40)
            # DATA LINE ****************

            strcenter(row=8, pos=20, txt=transl("p10t4"), width=40, size=1)
            strcenter(row=10, pos=20, txt="*........................", width=40, size=0)
            strcenter(row=15, pos=20, txt=transl("p10t5"), width=40, size=1)
            strcenter(row=17, pos=20, txt="*", width=40, size=0)
            # m._print('' + strformat(left=transl("p10t4")[:11], right="*........................", width=39))
            # m.pos(9)
            # m._print('' + strformat(left=transl("p10t5")[:11], right="*........................", width=39))
            # m.pos(11)
            # FINAL LINE ***************
            m.pos(21)
            m.color(m.bleu)
            m.plot('̶', 40)
            m.pos(22, 1)
            m._print(transl("p10t17"))
            m.pos(23, 1)

            m.color(m.vert)
            m._print(transl("p10t18"))
            m.pos(23, 12)
            m.inverse()
            m.color(m.cyan)
            m.underline()
            m._print("_SUITE ")

            m.pos(24, 1)
            m.color(m.vert)
            m._print(transl("p10t19"))
            m.pos(24, 12)
            m.inverse()
            m.color(m.cyan)
            m._print(" RETOUR")
            m.pos(24, 25)
            m._print(transl("p10t21"))
            m.pos(24, 33)
            m.inverse()
            m.color(m.cyan)
            m._print("SOMMAIRE")
            m.pos(22, 21)
            m._print(transl("p10t20"))
            m.pos(22, 33)
            m.inverse()
            m.color(m.cyan)
            m._print('  ENVOI ')
            m.pos(24, 28)

            # gestion de la zone de saisie courante******
            (zone, touche) = m.waitzones(zone, sltime)
            if touche == 1:
                # CHECK Input
                if m.zones[0]['texte'] == "":
                    m.message(20, 15, 3, "Name required")
                    break
                elif m.zones[1]['texte'] == "":
                    m.message(20, 15, 3, "Sexe required")
                    break
                elif m.zones[1]['texte'] not in ["m", "M", "h", "H", "f", "F", "w", "W", "d", "D"]:
                    m.message(20, 5, 3, "Sexe only M, H, F, W allowed")
                    break
                else:  # All ok
                    break
            if touche == 6:  # Sommaire
                break
            if touche == 5:
                break

        if touche == 1:
            if not [m.zones[0]['texte']] == "" or m.zones[1]['texte'] == "":
                data_p = [m.zones[0]['texte']]
                if m.zones[1]['texte'] in ["m", "M", "h", "H"]:
                    data_gender = "The questioner is a man"
                elif m.zones[1]['texte'] in ["f", "F", "w", "W"]:
                    data_gender = "The questioner is a woman"
                elif m.zones[1]['texte'] in ["d", "D"]:
                    data_gender = "Use gender-neutral language"
                else:
                    return
                m.resetzones()
                self.changeState(self.stateEnterData2)

        if touche == 6:  # Sommaire
            m.resetzones()
            self.changeState(self.stateWelcome)

    def stateEnterData2(self, entering):  # Enter Birthday and hour
        print("State Enter Birthday / Hour")
        m.canblock(6, 20, 1)
        global lang, sltime, data_p, data
        zone = 1
        m.zone(10, 11, 2, '', m.blanc)
        m.zone(10, 20, 2, '', m.blanc)
        m.zone(10, 30, 4, '', m.blanc)
        m.zone(19, 15, 2, '', m.blanc)
        m.zone(19, 28, 2, '', m.blanc)
        while True:
            # DATA LINE ****************

            strcenter(row=8, pos=20, txt=transl("p10t6"), width=40, size=1)
            # strcenter(row=10, pos=20, txt=(transl("p10t7") + " .. " + transl("p10t8") + " .. " + transl("p10t9") + " .... "), width=40, size=0)
            m.pos(10, 4)
            m._print(transl("p10t7") + "  ..")
            m.pos(10, 14)
            m._print(transl("p10t8") + " ..")
            m.pos(10, 24)
            m._print(transl("p10t9") + " ....")
            m.pos(13)

            strcenter(row=15, pos=20, txt=transl("p10t10"), width=40, size=1)
            strcenter(row=16, pos=20, txt=transl("p10t12a"), width=40, size=0)

            # strcenter(row=18, pos=20, txt = transl("p10t11") + " .. " + transl("p10t12") + " .. ", width=40, size=0)
            m.pos(19, 8)
            m._print(transl("p10t11") + " ..")
            m.pos(19, 19)
            m._print(transl("p10t12") + " ....")
            # m._print('' + strformat(left=transl("p10t10")[:6], right=transl("p10t11")+" .. "+transl("p10t12")+" .. ", width=39))
            # strcenter(row=16, pos=20, txt=transl("p10t12a"), width=40, size=0)
            # gestion de la zone de saisie courante******
            (zone, touche) = m.waitzones(zone, sltime)
            if touche == 1:
                break
            if touche == 6:  # Sommaire
                break
            if touche == 5:
                break
        if touche == 1:
            # CHECK Input
            n = ["day", "month", "year", "hours", "minutes"]
            for x in range(0, 4):
                if m.zones[x]['texte'] == "":
                    m.message(20, 15, 3, n[x] + " required")
                    return
            for x in range(0, 4):
                # print("Range x:", m.zones[x]['texte'])
                if not m.zones[x]['texte'].isdigit():
                    m.message(20, 7, 3, "Only digits allowed in " + n[x])
                    return
            if not int(m.zones[0]['texte']) in range(1, 32):
                m.message(20, 7, 3, "Not in range " + n[0])
                return
            if not int(m.zones[1]['texte']) in range(1, 13):
                m.message(20, 7, 3, "Not in range " + n[1])
                return
            if not int(m.zones[2]['texte']) in range(1800, 2400):
                m.message(20, 7, 3, "Not in range " + n[2])
                return
            if not int(m.zones[3]['texte']) in range(0, 24):
                m.message(20, 7, 3, "Not in range " + n[3])
                return
            if not int(m.zones[4]['texte']) in range(0, 60):
                m.message(20, 7, 3, "Not in range " + n[4])
                return

            data_p.extend(
                [int(m.zones[2]['texte']), int(m.zones[1]['texte']),
                 int(m.zones[0]['texte']), int(m.zones[3]['texte']),
                 int(m.zones[4]['texte'])]
            )
            # print(" + Datum DATA_p: ", data_p)
            m.resetzones()
            self.changeState(self.stateEnterData3)
        if touche == 6:  # Sommaire
            m.resetzones()
            self.changeState(self.stateWelcome)

    def stateEnterData3(self, entering):  # Enter Town and country
        print("State Enter Town / Country")
        m.canblock(6, 20, 1)
        global lang, sltime, data_p, data_astro, data, data_gender, kanye, language
        zone = 1
        m.zone(10, 8, 23, '', m.blanc)
        m.zone(15, 19, 2, '', m.blanc)
        m.zone(20, 19, 2, '', m.blanc)
        while True:
            # DATA LINE ****************
            strcenter(row=8, pos=20, txt=transl("p10t13"), width=40, size=1)
            strcenter(row=10, pos=20, txt="*........................", width=40, size=0)
            strcenter(row=13, pos=20, txt=transl("p10t14"), width=40, size=1)
            strcenter(row=14, pos=20, txt=transl("p10t16"), width=20, size=0)
            strcenter(row=15, pos=20, txt="*.", width=40, size=0)
            strcenter(row=18, pos=20, txt=transl("p10t15"), width=20, size=1)
            strcenter(row=19, pos=20, txt=transl("p10t16a"), width=20, size=0)
            strcenter(row=20, pos=20, txt="*.", width=40, size=0)

            # gestion de la zone de saisie courante******
            (zone, touche) = m.waitzones(zone, sltime)
            if touche == 1:
                break
            if touche == 6:  # Sommaire
                break
            if touche == 5:
                break
        if touche == 1:
            print("CHECK Input")
            n = ["Place of Birth", "Country of Birth", "Language to answer"]
            for x in range(0, 1):
                if m.zones[x]['texte'] == "":
                    m.message(21, 15, 3, n[x] + " required")
                    return
                if m.zones[2]['texte'] not in ["FR", "DE", "EN", "ES", "IT"]:
                    m.message(21, 15, 3, n[2] + " not in list")
                    return
                data_p.extend([m.zones[0]['texte'], m.zones[1]['texte']])
                lang_answ = get_language_by_country_code(m.zones[2]['texte'])
                # *****************************************************************************
                dat_ast = [data_p[0], data_p[1], data_p[2], data_p[3], data_p[4], data_p[5], data_p[6], data_p[7]]
                get_data_astro(dat_ast, str(m.zones[2]['texte']))
                # thread_1 = Thread(target=get_data_astro, args=(dat_ast, str(m.zones[2]['texte']))) # get_data_astro(dat_ast)
                # thread_2 = Thread(target=self.changeState, args=(self.stateWaitForAnswer1,))
                # thread_1.start()
                # time.sleep(2)
                # thread_2.start()
                # while thread_1.is_alive():
                # print("wait")
                # kanye = AstrologicalSubject("Sebastian", 1966, 12, 27, 17, 45,"Bielefeld", "DE", "",)
                # language = get_language_by_country_code(str(m.zones[2]['texte']))
                data_astro = ("Answer in " + language +
                              ", " + data_gender +
                              ", Astrological data: " + str(kanye) +
                              ", Sign: " + signs(kanye.sun['sign']) +
                              ", Ascendent: " + signs(kanye.first_house['sign']) +
                              ", Element Sun: " + kanye.sun.element +
                              ", Element Moon: " + kanye.moon.element +
                              ", Use also numerological Data."
                              )
                print(data_astro)
                m.resetzones()
                self.changeState(self.stateEnterQuest)
        if touche == 6:  # Sommaire
            m.resetzones()
            self.changeState(self.stateWelcome)

    def stateWaitForAnswer1(self, entering):
        global answer, message, thread_1, relay, data
        print("State Wait")
        answer = ""
        usbRelaysCheck()
        # while answer == "":
        y = False
        while thread_1.is_alive():

            for x in range(0, 200):
                m.pos(10, 18)
                m.scale(3)
                m._print(str(x))
                time.sleep(1)
                if not thread_1.is_alive():
                    if relay:
                        relay.set_state(1, False)
                    break

        self.changeState(self.stateSend)

    # ****************************************  Extra DATA (Question

    def stateEnterQuest(self, entering):

        global message, data_p, answer, thread_1, data_astro, data
        m.home()
        m.resetzones()
        touche = 0
        m.pos(4, 2)
        m.scale(3)
        m._print("Ta question?")

        # m.zone(ligne, colonne, longueur, texte, couleur)
        m.zone(7, 2, 38, "", m.blanc)
        m.zone(9, 2, 38, "", m.blanc)
        m.zone(11, 2, 38, "", m.blanc)
        m.zone(13, 2, 38, "", m.blanc)
        m.zone(15, 2, 38, "", m.blanc)
        m.zone(17, 2, 38, "", m.blanc)
        m.zone(19, 2, 38, "", m.blanc)
        m.zone(21, 2, 38, "", m.blanc)

        while True:
            x = 7
            while x <= 21:
                if not x % 2 == 0:
                    # print(x)
                    m.pos(x, 2)
                    m.plot('.', 37)
                x = x + 1
            zone = 1
            (zone, touche) = m.waitzones(zone, sltime)

            if touche == 1:
                break
            if touche == 6:
                break
        if touche == 1:

            mess = str("")
            for y in range(8):
                if not m.zones[y]['texte'] == "":
                    mess = str(mess) + " Extra text: " + str(m.zones[y]['texte'] + " ")
            print(mess)
            message = data_astro + mess

            m.home()  # TODO link to waitingpage while processing quest

            thread_1 = Thread(target=chatbot, args=(message,))
            # thread_2 = Thread(target=self.changeState, args=(self.stateWaitForAnswer2,))
            thread_1.start()
            # time.sleep(2)
            # thread_2.start()
            #
            # chatbot(message)
            # self.changeState(self.stateSend)
            self.changeState(self.stateWaitForAnswer2)
        if touche == m.sommaire and zone == "sleep":
            # choix1 = ""
            print("Go to sleeper")
            self.changeState(self.stateWelcome)
        if touche == 6:  # Sommaire
            m.resetzones()
            self.changeState(self.stateWelcome)

    # ****************************************
    def stateWaitForAnswer2(self, entering):
        global answer, message, thread_1, data
        print("State Wait2")
        answer = ""
        r = usbRelaysCheck()
        print("What says relais?", relay.state)

        # while answer == "":
        y = False
        while thread_1.is_alive():
            if r:
                print("Relais to true")
                relay.set_state(1, True)
                # relay.set_state(4, True)
            for x in range(0, 200):
                m.pos(10, 18)
                m.scale(3)
                m._print(str(x))
                time.sleep(1)
                if x > 2 and x < 5:
                    relay.set_state(4, True)
                else:
                    relay.set_state(4, False)

                if not thread_1.is_alive():
                    if relay:
                        relay.set_state(1, False)
                    break

        self.changeState(self.stateSend)

    # ****************************************
    def stateSend(self, entering):
        global message, p, pV, answer, lang, data, data_p

        # answer = chatbot(message)

        answer = answer.replace("œ", "oe")
        answer = answer.replace("Œ", "Oe")
        print("State Send")
        # print(data_p)
        # pV = printCheck()
        # *****prepa array
        # with open("test.txt", "r") as file:
        # file_content = file.read()
        file_content = answer
        lines = split_string_into_lines(file_content, max_line_length=39)
        print_lines = split_string_into_lines(file_content, max_line_length=32)

        print("Anzahl Zeilen", len(lines))  # anzahl zeilen
        # print(pV)
        m.home()
        m.pos(1, 1)
        page = 1

        # (choix, touche) = m.input(21, 29, 1, sltime)
        # else:
        # (choix, touche) = m.input(21, 1, 0, sltime)
        # plusieurs pages ?
        if len(lines) > 18:
            m.pos(0, 33)
            m._print(" " + str(int(abs(page))) + '/' + str(int((len(lines) + 17) / 18)))

        while True:
            m.home()
            pV = printCheck()
            if len(lines) > 0:  # affichage
                # entête sur 2 lignes + séparation
                m.pos(0)
                m._print("TEST3")  # transl("p5t1"))
                if lines != '':
                    # m.pos(0, 36)
                    m.color(m.bleu)
                    m._print(str(len(lines)))
                m.pos(2)
                m.color(m.bleu)
                m.plot('̶', 40)

                # plusieurs pages ?
                if len(lines) > 18:
                    m.pos(0, 33)
                    m._print(" " + str(int(abs(page))) + '/' + str(int((len(lines) + 17) / 18)))
                    m.pos(3)

                # première ligne de résultat
                y = 3
                m.pos(y)
                for a in range((page - 1) * 18, page * 18, ):
                    # for a in range( page*9,(page-1)*9, -1): # neu rückwärts
                    # print("Anzeige page:", (page - 1) * 18, "  ", page * 18)
                    y = y
                    m.pos(y, 1)
                    if a < len(lines):
                        r = lines[a]
                        # print(r, len(r))
                        m.color(m.blanc)
                        strcenter(row=y, pos=20, txt=r, width=40, size=0)
                        # m._print(r)
                        y = y + 1
                        m.pos(y, 1)
                        m.color(m.vert)
                        m.color(m.bleu)

                # ligne finale
                m.pos(21)
                m.color(m.bleu)
                m.plot('̶', 40)

                if page > 1:
                    if len(lines) > page * 18:  # place pour le SUITE
                        m.pos(22, 22)
                    else:
                        m.pos(22, 22)
                    m.color(m.vert)
                    m._print("precedent")
                    m.pos(22, 33)
                    m.underline()
                    m._print(' ')
                    m.inverse()
                    m.color(m.cyan)
                    m._print('_RETOUR')

                if len(lines) > page * 18:
                    m.pos(23, 21)
                    m.color(m.vert)
                    m._print("suivante")
                    m.pos(23, 34)
                    m.underline()
                    m._print(' ')
                    m.inverse()
                    m.color(m.cyan)
                    m._print('_SUITE')
                pV = printCheck()

                if pV == True:
                    m.pos(22, 1)
                    m._print("PRINT Y/N   + → ")
                    m.pos(22, 16)
                    m.inverse()
                    m.color(m.cyan)
                    m._print("ENVOI")
                    m.cursor(True)
                m.pos(23, 5)

                m.pos(24, 1)
                m.color(m.vert)
                m._print("Aide")
                m.pos(24, 11)
                m.inverse()
                m.color(m.cyan)
                m._print(' GUIDE')
                m.underline()
                m._print(' ')
                m.inverse()
                m.color(m.cyan)
                m.pos(24, 26)
                m.color(m.vert)
                m._print("Home")
                m.pos(24, 33)
                m.inverse()
                m.color(m.cyan)
                m._print("SOMMAIRE")

            else:
                page = abs(page)

                # attente saisie            
            if pV == True:
                (choix, touche) = m.input(22, 11, 1, sltime)
            else:
                (choix, touche) = m.input(22, 1, 1, sltime)
                m.cursor(False)
            # ****** check for integer
            if choix == "y" or choix == "Y":
                choix1 = "Y"
            else:
                choix1 = ""
            if not isinstance(choix, int):
                if not choix.isdigit():
                    choix = 0
                choix = int(choix)

            if choix == "":
                choix = int(0)

            elif choix > len(lines):
                break
            else:
                choix = int(choix)
            m.cursor(False)
            if touche == m.suite:
                if page * 18 < len(lines):
                    page = page + 1
                else:
                    m.bip()
            elif touche == m.retour:
                if page > 1:
                    page = page - 1
                else:
                    m.bip()
            elif touche == m.envoi and choix1 == "Y":
                # print("angekommen")
                z = random.randint(1, 24)
                # print("./pics/code_" + str(z) + ".png")
                if pV == True:
                    p.text("\n")
                    p.image("./WM/ww.png")
                    p.text("\n")
                    p.image("./WM/fortune.png")
                    p.set(font='a', height=2, width=1, align='center')  # , text_type='bold')
                    p.text("\n \n")
                    p.set(font='a', height=1, width=1, align='center')  # , text_type='bold')
                    try:
                        for b in range(0, len(print_lines)):
                            x = print_lines[b]
                            p.text(str(x) + "\n")
                    except:
                        p.text("Can't print this language, \n sorry \n ")
                        # print(b)
                    p.text("\n More informations or \n")
                    p.text("share your experience\n")
                    p.text("check the QR code!\n")
                    # p.qr('WISH WIZARD PROJECT \n ' + answer, 0, 5, 2)
                    p.qr('WISH WIZARD PROJECT \n https://www.facebook.com/wishwizardproject', 0, 5, 2)
                    # p.set(font='b', height=2, width=2, align='center') #, text_type='bold')
                    p.text("_________________________")
                    p.text("\n \n")
                    p.set(align='center')
                    p.image("./WM/pics/code_" + str(z) + ".png")
                    p.text("\n")
                    p.text("_________________________")
                    p.cut()
                    p.close()
                break
            elif touche == m.annulation:
                break
            elif touche == m.guide:
                break
            elif touche == m.sommaire:
                break
            elif touche == m.correction:  # retour saisie pour correction
                break
                # return(touche)
            elif touche == m.repetition:
                # print(print_lines)
                break

                # end while

        if touche == m.sommaire and choix == "sleep":
            # choix1 = ""
            print("Go to sleeper")
            m.resetzones()
            self.changeState(self.stateWelcome)
        if touche == m.sommaire:
            m.resetzones()
            self.changeState(self.stateWelcome)


####
# Program entry point
###
def main():
    global data, ver

    # Define Version
    ver = "0.1"
    print("Fortune Teller ", ver, "\n")
    # create state machine object
    stateMachine = StateMachine()

    # Create settings.ini if not exist
    ini_exists = os.path.isfile("settings.ini")
    # print("INI file exist? ", ini_exists)
    if not ini_exists:
        config = configparser.ConfigParser()
        config['prefs'] = {
            'IP_adr': '127.0.0.1',
            'speed': '1200',
            'timer': '120',
            'lang_code': 'FR'
        }
        config['printer'] = {
            'p_idvend': '0x04b8',
            'p_idprod': '0x0e1f',
            'p_timer': '',
            'p_3': '',
            'p_4': ''}
        config['IA'] = {
            'api_key': '',
            'ia_model': 'gpt-3.5-turbo',
            'max_tok': '300'}
        with open('settings.ini', 'w') as configfile:
            config.write(configfile)
        print("Ini file created")
    config = configparser.ConfigParser()
    config.read('settings.ini')

    # Prepare dataobject with all Screen txts in diff languages
    csv_dateipfad = './WM/lang.csv'
    data = create_data(csv_dateipfad)  # to function

    # Prepa divers
    global m
    if len(sys.argv) > 2:
        (choice, ou) = (sys.argv[1], sys.argv[2])
    else:
        (choice, ou) = ('', '')

    print(" ")
    ###
    # Main loop
    ###
    while True:
        stateMachine.update()


# start
if __name__ == '__main__':
    main()

exit()
