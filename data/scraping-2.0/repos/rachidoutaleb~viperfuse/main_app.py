import json
import subprocess
import threading
import time
import tkinter as tk
import customtkinter as ctk
import cv2
import os
import speech_recognition as sr
import nmap
import openai
import pyfiglet
from datetime import datetime
from tkinter import messagebox
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QApplication, QCompleter, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, \
    QTextEdit, QVBoxLayout, QWidget, QToolButton
from cryptography.fernet import Fernet
from fuzzywuzzy import fuzz
from scapy.all import ARP, Ether, srp
import qdarktheme
import face_recognition
import pickle
import re
from gtts import gTTS
from playsound import playsound


BASE_DIR = "face_encodings"
ICON_PATH = 'logo2.png'
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

ctk.set_appearance_mode("System")
r = sr.Recognizer()

#check si utilisateur demande d'executer un programe
def check_if_execute(input: str)->str:
    split_str = input.rsplit()
    x = check_if_in_string(split_str, "execute") | check_if_in_string(split_str, "start") | check_if_in_string(split_str, "open")
    if not x:
        return ''
    net_scan = check_if_in_string(split_str, "networ") & check_if_in_string(split_str, "analyzer")
    port_scan = check_if_in_string(split_str, "port") & check_if_in_string(split_str, "scann")
    ip_scan = check_if_in_string(split_str, "ip") & check_if_in_string(split_str, "scann")
    file_intg = check_if_in_string(split_str, "file") & check_if_in_string(split_str, "integrity")
    if net_scan: return "open network analyzer"
    if ip_scan: return  "open ip scanner"
    if port_scan: return  "open scan port"
    if file_intg: return "open file integrity"
    else:
        return "You didn't specify which program !"

def execute_script(path="portscanner.py"):
    script_path = path
    result = subprocess.run(["python", script_path], capture_output=True, text=True)

    if result.returncode == 0:
        print("Script executed successfully.")
    else:
        print("Script execution failed.")
        print("Error message:", result.stderr)

#Detect face and draw box
def detect_bounding_box(vid,face_classifier=face_classifier):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

def register_face(email):
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)
#
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if frame is not None and frame.size > 0:
            faces = detect_bounding_box(frame)
            cv2.imshow("Register Your Face", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('\n') or key == ord('\r'):
                captured_frame = frame.copy()
                break
        else:
            print("Error: Frame not captured or empty.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
        face_encodings = face_recognition.face_encodings(captured_frame)

        if face_encodings:
            encoding_path = os.path.join(BASE_DIR, f"{email}.pkl")
            with open(encoding_path, "wb") as f:
                pickle.dump(face_encodings[0], f)

            return "Face encoding saved."
        else:
            return "No face detected. Please try again."
    else:
        return "No frame captured. Please check the camera connection."

def login_face(email):
    encoding_path = os.path.join(BASE_DIR, f"{email}.pkl")

    if not os.path.exists(encoding_path):
        return "No encoding found for this email. Please register first."

    with open(encoding_path, "rb") as f:
        known_encoding = pickle.load(f)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if frame is not None and frame.size > 0:
            faces = detect_bounding_box(frame)
            cv2.imshow("Login with Your Face", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('\n') or key == ord('\r'):
                captured_frame = frame.copy()
                break
        else:
            print("Error: Frame not captured or empty.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
        face_encodings = face_recognition.face_encodings(captured_frame)

        if face_encodings:
            matches = face_recognition.compare_faces([known_encoding], face_encodings[0])

            if matches[0]:
                return "Face authenticated. You're logged in!"
            else:
                return "Face not recognized. Please try again."
        else:
            return "No face detected. Please try again."
    else:
        return "No frame captured. Please check the camera connection."
    



class FaceRecognitionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("ViperFuse")
        self.root.geometry("400x300")

        self.action_var = ctk.StringVar(value='login')
        self.email_var = ctk.StringVar()

        self.action_choices = ['login', 'register']
        self.app_closed = False

        self.create_widgets()
        #to skip face auth
        self.open_chatbot_ui()

        return

    def create_widgets(self):
        ctk.CTkLabel(self.root, text="Welcome To ViperFuse", font=("Helvetica", 24)).pack(pady=20)

        action_frame = ctk.CTkFrame(self.root,fg_color='transparent')
        action_frame.pack(pady=10)

        ctk.CTkLabel(action_frame, text="Choose action: ").pack(side=tk.LEFT)
        self.action_menu = ctk.CTkOptionMenu(action_frame, variable=self.action_var, values=self.action_choices,width=90)
        #self.action_menu.pack(side=tk.LEFT)

        #
        self.segemented_button = ctk.CTkSegmentedButton(master=action_frame,
                                                             values=self.action_choices,
                                                             variable=self.action_var)
        self.segemented_button.pack(padx=20, pady=10,side=tk.LEFT)
        #
        email_frame = ctk.CTkFrame(self.root,fg_color='transparent')
        email_frame.pack(pady=10)

        ctk.CTkLabel(email_frame, text="Enter email: ").pack(side=tk.LEFT)
        email_entry = ctk.CTkEntry(email_frame, textvariable=self.email_var,width=200)
        email_entry.pack(side=tk.LEFT)

        button_frame = ctk.CTkFrame(self.root)
        button_frame.pack(pady=10)

        register_button = ctk.CTkButton(button_frame, text="Proceed", command=self.process_action)
        register_button.pack(side=tk.LEFT)




    def open_chatbot_ui(self):
        chatbot_app = QApplication([])
        qdarktheme.setup_theme()  #dark theme
        chatbot_window = ChatbotUI()
        chatbot_window.show()
        chatbot_app.exec_()


    def process_action(self):
        action = self.action_var.get().lower()
        email = self.email_var.get()

        if action == "register":
            result = register_face(email)
            if result == "Face encoding saved.":
                messagebox.showinfo("Success", result)
            else:
                messagebox.showerror("Error", result)
        elif action == "login":
            result = login_face(email)
            if result == "Face authenticated. You're logged in!":
                self.close_app()
                self.open_chatbot_ui()
            else:
                messagebox.showerror("Error", result)
        else:
            messagebox.showerror("Error", "Invalid action. Please choose 'login' or 'register'.")

    def close_app(self):
        self.app_closed = True
        self.root.destroy()

    def check_app_closed(self):
        if self.app_closed:
            execute_script()
        else:
            self.root.after(1000, self.check_app_closed)

def check_if_in_string(input: list, word: str):
    splt = input
    for i in splt:
        if fuzz.ratio(word.lower(), i.lower()) > 70:
            return True
    return False

def check_and_close(app):
    while True:
        if not app.is_alive():
            execute_script()
            break
        time.sleep(1)










class ChatbotUI(QMainWindow):

    openai.api_key = "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"
    
    def __init__(self):
        super().__init__()

        with open("intentions.json", "r") as file:
            self.intentions = json.load(file)["intentions"]

        self.current_process = None
        self.init_ui()
        self.setWindowIcon(QtGui.QIcon(ICON_PATH)) #logo

            
    def send_message(self):
        user_message = self.user_input.text()
        response_found = False
        bot_response = "Je n'ai pas compris votre demande. Veuillez reformuler."

        #check si utiliosateur demande d'excuter
        _check = check_if_execute(user_message)
        user_message = _check if (_check != '') else  user_message

        # List of commands to check for
        commands = ["ls", "cat", "mkdir", "grep", "tr", "cut", "sed", "cp", "mv"]
        commands += ["dir" ," type" , "mkdir" ,"cp" , "mv"]

        # Check if the user message starts with a command
        for command in commands:
            if user_message.lower().startswith(command):
                try:
                    # The command is the first word, the rest are arguments
                    command_args = user_message.split(" ")
                    result = subprocess.run(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    bot_response = result.stdout.decode()
                    if result.stderr:
                        bot_response += "\nErrors:\n" + result.stderr.decode()
                    response_found = True
                except Exception as e:
                    bot_response = "Error executing command: " + str(e)
                    response_found = True
        if user_message:
            self.chat_text.append("User: " + user_message)
            self.user_input.clear()

            # une fonction pour recherchez intention correspondante
            def find_best_response_from_file(input,intentions)->tuple[bool,str]:
                possible_rep: list[tuple[int, str, str]] = []
                # check possible rep
                for intent in intentions:
                    for phrase in intent["patterns"]:
                        perc = fuzz.ratio(phrase.lower(), input.lower())
                        if perc > 65:
                            possible_rep.append((perc, phrase, intent['response']))
                # choose rep
                if len(possible_rep) > 0:
                    best_rep = max(possible_rep, key=lambda x: x[0])  # rep with highest percenatge
                    response_found = True
                    bot_response = best_rep[2]
                else:
                    response_found = False
                    bot_response = ''
                #debug information :
                possible_rep.sort(key=lambda x: x[0])
                for x in possible_rep:pass
                    #print(x)
                #
                return response_found,bot_response

            response_found, bot_response = find_best_response_from_file(user_message,self.intentions)


            if "team" in user_message.lower():
                bot_response = "Mahmoud Charif\nRachid Outaleb\nIsmail Ouzaid\nZineb "
                response_found = True
                
            # Check for clear command
            elif user_message.lower() == "clear":
                self.clear_chat()
                return    


            elif fuzz.ratio("open scan port", user_message.lower()) > 70:  
                bot_response = "Ouverture de ScanPort..."
                response_found = True
                self.type_message("AI: ", bot_response)
                QTimer.singleShot(5000, self.open_port_scanner)  
                return 
            elif fuzz.ratio("open ip scanner", user_message.lower()) > 70:
                bot_response = "Ouverture de IP Scanner..."
                response_found = True
                self.type_message("AI: ", bot_response)
                QTimer.singleShot(5000, self.open_network_scanner)  
                return
            elif fuzz.ratio("open file integrity", user_message.lower()) > 70:
                bot_response = "Ouverture de File integrity ..."
                response_found = True
                self.type_message("AI: ", bot_response)
                QTimer.singleShot(5000, self.open_file_integrity)
                return
            elif fuzz.ratio("open network analyzer", user_message.lower()) > 70:
                bot_response = "Ouverture du NetAnalyzer..."
                response_found = True
                self.type_message("AI: ", bot_response)
                QTimer.singleShot(5000, self.open_network_analyzer)
                return
            elif fuzz.ratio("You didn't specify which program !", user_message.lower()) >80:
                user_message = ""
                bot_response = "You didn't specify which program !"
                response_found = True
            elif fuzz.ratio("save discussion", user_message.lower()) > 70:
                bot_response = "Sauvegarde de la discussion..."
                response_found = True
                self.type_message("AI: ", bot_response)
                QTimer.singleShot(2000, self.save_conversation)
                return 
            elif fuzz.ratio("display discussion", user_message.lower()) > 70:
                bot_response = "Affichage de la discussion..."
                response_found = True
                self.type_message("AI: ", bot_response)
                QTimer.singleShot(2000, self.load_conversation)
                return 

            # Si aucune intention n'a été trouvée, utilisez l'API GPT-3
            if not response_found:

                #delete


                prompt = f"User: {user_message}\nAI:"
                response = "?"
                #TODO delete
                #response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=50, n=1, stop=None, temperature=0.5)
                #bot_response = response.choices[0].text.strip()

            self.type_message("AI: ", bot_response)
            return bot_response
     
     
    def clear_chat(self):
        self.chat_text.clear() 
            
    def execute_option_script(self, option):
        script_paths = {
            "portscanner": "portscanner.py",
            "ip scanner": "networkscanner.py",
            "network analyzer": "Network_analyzer.py"
        }

        if option in script_paths:
            script_path = script_paths[option]
            result = subprocess.run(["python3", script_path], capture_output=True, text=True)

            if result.returncode == 0:
                print(f"{option} script executed successfully.")
            else:
                print(f"{option} script execution failed.")
                print("Error message:", result.stderr)
        else:
            print("Invalid option.")

    def open_network_scanner(self):
        self.network_scanner = NetworkScanner()
        self.network_scanner.show()
        self.hide()


    def init_ui(self):
        self.setWindowTitle('ChatFuse')

        widget = QWidget()
        layout = QVBoxLayout()

        self.chat_label = QLabel('Chat:')
        layout.addWidget(self.chat_label)

        self.chat_text = QTextEdit()
        self.chat_text.setReadOnly(True)
        layout.addWidget(self.chat_text)

        self.user_input_label = QLabel('Your message:')
        layout.addWidget(self.user_input_label)

        input_layout = QHBoxLayout()

        self.user_input = QLineEdit()
        input_layout.addWidget(self.user_input)

        #auto completer part
        self.model = QStandardItemModel()  #fill it with self.model.appendRow(QStandardItem(entryItem))
        completer = QCompleter(self.model, self)
        #completer.setCompletionMode()   # InlineCompletion / UnfilteredPopupCompletion / PopupCompletion
        for intent in self.intentions:
            for pattern in intent['patterns']:
                #print(pattern)
                self.model.appendRow(QStandardItem(pattern.lower()))
        self.user_input.setCompleter(completer)

        #

        self.send_button = QPushButton('Send')
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        

        layout.addLayout(input_layout)

        # speech recognition button
        self.speak_button = QPushButton('Speak')
        self.speak_button.clicked.connect(self.speak_message)
        # layout.addWidget(self.speak_button)

        self.stop_speak_button = QPushButton('Stop Speak')
        self.stop_speak_button.clicked.connect(self.stop_speak)
        # layout.addWidget(self.stop_speak_button)

        self.speech_layout = QHBoxLayout()
        for btn in (self.speak_button,self.stop_speak_button):
            self.speech_layout.addWidget(btn)

        layout.addLayout(self.speech_layout)

        #testing
        self.tst_button = QPushButton('')
        # self.nxt_button = QPushButton('next')
        # self.nxt_button.clicked.connect(self.change_button)
        self.tst_layout = QHBoxLayout()
        buttonL = QToolButton()
        buttonL.clicked.connect(lambda : self.change_button('L'))
        buttonL.setArrowType(QtCore.Qt.LeftArrow)
        buttonR = QToolButton()
        buttonR.clicked.connect(lambda : self.change_button('R'))
        buttonR.setArrowType(QtCore.Qt.RightArrow)

        for btn in (buttonL,self.tst_button,buttonR):
            self.tst_layout.addWidget(btn)
        layout.addLayout(self.tst_layout)
        self.change_button()
        #


        widget.setLayout(layout)
        self.setCentralWidget(widget)
       

        # Créez le menu d'options
        
        '''
        # Créez les actions de menu pour les options
        self.option1_button = QPushButton('Port Scanner')
        self.option1_button.clicked.connect(self.open_port_scanner)
        layout.addWidget(self.option1_button)

        self.option2_button = QPushButton('IP scanner')
        self.option2_button.clicked.connect(self.open_network_scanner)
        layout.addWidget(self.option2_button)
        
        
        self.network_analyzer_button = QPushButton('Network Analyzer')
        self.network_analyzer_button.clicked.connect(self.open_network_analyzer)
        layout.addWidget(self.network_analyzer_button)

        self.File_Integrity_button = QPushButton('File Integrity')
        self.File_Integrity_button.clicked.connect(self.open_file_integrity)
        layout.addWidget(self.File_Integrity_button)
        '''

        # Ajoutez le bouton Save
        self.save_button = QPushButton('Save Conversation')
        self.save_button.clicked.connect(self.save_conversation)
        layout.addWidget(self.save_button)

        # Ajoutez le bouton Load
        self.load_button = QPushButton('Load Conversation')
        self.load_button.clicked.connect(self.load_conversation)
        layout.addWidget(self.load_button)
        
        self.user_input.returnPressed.connect(self.send_message)

    tst_button_id = 0 #  0-4
    def change_button(self,id=None):
        #self.statusBar().showMessage(f'button {id} was pressed')
        ls = [
            ["Port scanner","Network Analyzer","Ip scanner","File integrity"],
            [self.open_port_scanner,self.open_network_analyzer,self.open_network_scanner,self.open_file_integrity]
        ]
        def set_button(id,ls=ls):
            try:
                self.tst_button.clicked.disconnect()
            except:pass
            self.tst_button.clicked.connect(ls[1][id])
            self.tst_button.setText(ls[0][id])
        if id == 'R':
            self.tst_button_id+=1
            if self.tst_button_id >= 4: self.tst_button_id-=4
        if id == 'L':
            self.tst_button_id -= 1
            if self.tst_button_id < 0: self.tst_button_id +=4
        else:pass
        set_button(self.tst_button_id,ls)
    def open_port_scanner(self):
        self.hide()
        self.port_scanner = PortScanner(self)
        self.port_scanner.show()    
        
    """def send_message(self):
        user_message = self.user_input.text()

        if user_message:
            self.chat_text.append("User: " + user_message)
            self.user_input.clear()

            # Utiliser l'API GPT pour générer une réponse
            prompt = f"User: {user_message}\nAI:"
            #response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=50, n=1, stop=None, temperature=0.5)

            bot_response = response.choices[0].text.strip()
            self.type_message("AI: ", bot_response)"""

    #
    def speak_message(self):
        self.type_message("AI: ", "Listening...")
        def func():
            def get_voice_input():
                with sr.Microphone() as source:
                    audio = r.listen(source)
                    try:
                        text = r.recognize_google(audio)
                        return text
                    except:
                        return "Sorry could not recognize your voice"
                    
            text = get_voice_input()
            if text == "Sorry could not recognize your voice":
                response="Sorry could not recognize your voice"
                self.type_message("AI: ", response)
            else:
                # Obtenez une réponse du chatbot     
                self.user_input.setText(text)
                response = self.send_message()
                

            print(text)

            # Conversion de la réponse en voix
            speech = gTTS(text=str(response), lang='en', slow=False)
            speech.save("response.mp3")

            # Lecture de la réponse
            playsound(f"{os.path.dirname(__file__)}\\response.mp3")

        thr = threading.Thread(target=func)
        thr.start()


    def stop_speak(self):
        #recognizer.stop()
        pass

    def speak_message__old(self):
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        with microphone as source:
            self.type_message("AI: Listening...")
            self.chat_text.append("Listening...")
            audio = recognizer.listen(source)

        try:
            self.chat_text.append("Recognizing...")
            user_message = recognizer.recognize_google(audio)
            self.chat_text.append("User: " + user_message)

            # Utiliser l'API GPT pour générer une réponse
            prompt = f"User: {user_message}\nAI:"
            response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=50, n=1, stop=None, temperature=0.5)

            bot_response = response.choices[0].text.strip()
            self.type_message("AI: ", bot_response)
        except sr.UnknownValueError:
            self.chat_text.append("Could not understand audio")
        except sr.RequestError as e:
            self.chat_text.append(f"Error: {e}")
            
    

    def type_message(self,prefix, message):
        full_message = '\n' + prefix + message
        for i in range(len(full_message)):
            QTimer.singleShot(i * 50, lambda i=i: self.chat_text.insertPlainText(full_message[i]))
      
    def open_network_scanner(self):
        self.network_scanner = NetworkScanner(self)
        self.network_scanner.show()
        self.hide()

    def open_file_integrity(self):
        proc = subprocess.run(["python", "file_intg.py"])
    
    def open_network_analyzer(self):
        import Network_analyzer
        Network_analyzer.main()
    
    # Générer une clé de chiffrement Fernet et la sauvegarder dans un fichier
    def generate_key(self):
        key = Fernet.generate_key()
        with open("key.key", "wb") as key_file:
            key_file.write(key)

    # Charger la clé de chiffrement Fernet à partir d'un fichier
    def load_key(self):
        return open("key.key", "rb").read()

    # Chiffrer le texte en utilisant la clé Fernet
    def encrypt_text(self, text, key):
        f = Fernet(key)
        encrypted_text = f.encrypt(text.encode())
        return encrypted_text

    # Déchiffrer le texte en utilisant la clé Fernet
    def decrypt_text(self, encrypted_text, key):
        f = Fernet(key)
        decrypted_text = f.decrypt(encrypted_text).decode()
        return decrypted_text

    # Sauvegarder la conversation chiffrée dans un fichier
    def save_conversation(self):
        # Générer et charger la clé de chiffrement
        self.generate_key()
        key = self.load_key()
        
        # Chiffrer la conversation
        conversation = self.chat_text.toPlainText()
        encrypted_conversation = self.encrypt_text(conversation, key)

        # Sauvegarder la conversation chiffrée dans un fichier
        with open("conversation.enc", "wb") as file:
            file.write(encrypted_conversation)

        print("Conversation saved.")
        self.type_message("AI: ", "Conversation saved.")

    # Charger et afficher la conversation déchiffrée
    def load_conversation(self):
        # Charger la clé de chiffrement
        key = self.load_key()

        # Charger et déchiffrer la conversation
        with open("conversation.enc", "rb") as file:
            encrypted_conversation = file.read()

        conversation = self.decrypt_text(encrypted_conversation, key)

        # Afficher la conversation déchiffrée
        self.chat_text.setPlainText(conversation)

        print("Conversation loaded.")
        self.type_message("AI: ", "Conversation loaded.")
                   
class PortScanner(QMainWindow):
    def __init__(self, chatbot_ui):
        super().__init__()
        self.chatbot_ui = chatbot_ui
        self.init_ui()
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Port Scanner')

        widget = QWidget()
        layout = QVBoxLayout()

        self.target_label = QLabel('Target:')
        layout.addWidget(self.target_label)

        self.target_entry = QLineEdit()
        layout.addWidget(self.target_entry)

        self.scan_button = QPushButton('Scan')
        self.scan_button.clicked.connect(self.start_scan)
        layout.addWidget(self.scan_button)

        self.stop_button = QPushButton('Stop')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_scan)
        layout.addWidget(self.stop_button)
        self.back_button = QPushButton('Back')
        self.back_button.clicked.connect(self.go_back_to_chatbot)
        layout.addWidget(self.back_button)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.new_scan_button = QPushButton('New Scan')
        self.new_scan_button.setEnabled(False)
        self.new_scan_button.clicked.connect(self.new_scan)
        layout.addWidget(self.new_scan_button)

        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.thread = PortScannerThread()
        self.thread.result_signal.connect(self.handle_scan_result)

    def start_scan(self):
        target = self.target_entry.text()

        if target:
            ascii_banner = pyfiglet.figlet_format("PORT SCANNER")
            self.result_text.append(ascii_banner)

            self.result_text.append("-" * 50)
            self.result_text.append("Scanning Target: " + target)
            self.result_text.append("Scanning started at:" + str(datetime.now()))
            self.result_text.append("-" * 50)

            self.scan_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.new_scan_button.setEnabled(False)

            self.thread.set_target(target)
            self.thread.start()

    def stop_scan(self):
        self.thread.stop()

        self.scan_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.new_scan_button.setEnabled(True)

    def new_scan(self):
        self.thread.stop()
        self.target_entry.setText('')
        self.result_text.setText('')
        self.scan_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.new_scan_button.setEnabled(False)

    def handle_scan_result(self, port, status):
        if status == "open":
            self.result_text.append("Port {} is open".format(port))

    def closeEvent(self, event):
        self.thread.stop()
        self.thread.wait()
        event.accept()
    def go_back_to_chatbot(self):
        self.close()
        self.chatbot_ui.show()

    def closeEvent(self, event):
        self.thread.stop()
        self.thread.wait()
        event.accept()

class PortScannerThread(QThread):
    result_signal = pyqtSignal(int, str)

    def __init__(self):
        super().__init__()
        self._stop = False

    def set_target(self, target):
        self.target = target

    def run(self):

        #
        
        def extract_string_after_ip(user_input):
        # Find the IP address using regular expression
            ip_match = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', user_input)
            
            if ip_match:
                ip_address = ip_match.group(0)
                index = user_input.find(ip_address)
                
                # Extract the string after the IP address
                extracted_string = user_input[index + len(ip_address):].strip()
                
                return extracted_string,ip_address
        #

        extra_arg, self.target= extract_string_after_ip(self.target)

        nm = nmap.PortScanner()
        nm.scan(hosts=self.target, arguments='-p 1-65535 --open {extra_arg}')
        
        for host in nm.all_hosts():
            for proto in nm[host].all_protocols():
                lport = nm[host][proto].keys()
                for port in lport:
                    self.result_signal.emit(port, "open")

        self.stop()
        if self._stop:
            return

    def stop(self):
        self._stop = True
        
class NetworkScanner(QMainWindow):
    def __init__(self, chatbot_ui):
        super().__init__()
        self.chatbot_ui = chatbot_ui
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Network Scanner')

        widget = QWidget()
        layout = QVBoxLayout()

        self.target_label = QLabel('Target IP:')
        layout.addWidget(self.target_label)

        self.target_entry = QLineEdit()
        layout.addWidget(self.target_entry)

        self.scan_button = QPushButton('Scan')
        self.scan_button.clicked.connect(self.network_scan)
        layout.addWidget(self.scan_button)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.back_button = QPushButton('Back')
        self.back_button.clicked.connect(self.back_to_chatbot)
        layout.addWidget(self.back_button)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def network_scan(self):
        target_ip = self.target_entry.text()

        if target_ip:
            arp = ARP(pdst=target_ip)
            ether = Ether(dst="ff:ff:ff:ff:ff:ff")
            packet = ether/arp

            result = srp(packet, timeout=3, verbose=0)[0]

            clients = []

            for sent, received in result:
                clients.append({'ip': received.psrc, 'mac': received.hwsrc})

            self.result_text.append("Available devices in the network:")
            self.result_text.append("IP" + " "*18 + "MAC")
            for client in clients:
                self.result_text.append("{:16}    {}".format(client['ip'], client['mac']))
    
    def back_to_chatbot(self):
        self.hide()
        self.chatbot_ui.show()                

def main():
    root = ctk.CTk()
    app = FaceRecognitionApp(root)                  
    app.check_app_closed()
    root.mainloop()

if __name__ == "__main__":
    main()


