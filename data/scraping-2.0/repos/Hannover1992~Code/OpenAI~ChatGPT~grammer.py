import socket
from transformers import pipeline
from openai import OpenAI
import threading

# Initialisieren des GPT-3-Modells
client = OpenAI()

def generate_response(text, Instruction):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": Instruction},
            {"role": "user", "content": text}
        ]
    )

    # Extrahieren Sie den Text der Antwort auf die korrekte Weise
    # Hier wird angenommen, dass 'message' ein Attribut des Objekts ist
    response_text = completion.choices[0].message.content
    return response_text

def setup_server(port, instructon):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', port))
    server_socket.listen(1)
    print("Server listening on port" + str(port))

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        message = client_socket.recv(1024).decode()
        if message:
            print("Received message:", message)
            response = generate_response(message, instructon)
            with open('response.txt', 'w') as f:
                f.write(response)
            client_socket.sendall(response.encode())

            client_socket.close()

def main():

    instruction_Grammer = "Sie sind ein Sprachassistent, der darauf spezialisiert ist, deutsche Texte zu korrigieren und zu verbessern.Du solltes nur das korriegertes Satz ausgeben. Achte auf gramatik Zeichensetugn, satzbau Wortwahl. Gebe nur das verbesser satz zuruck ohne weitern text."

    instruction_Letter = "Die Aufgabe dieses Chats besteht darin, dir Anweisungen für das Verfassen eines Briefes, einer Nachricht oder einer E-Mail zu geben. Ich werde den Inhalt vorgeben, und du sollst die Nachricht für mich ausformulieren.Vergiss nicht das ich dich nur mit groben anweisen austate, dein aufgeb ist es immer so zu schreiben als hatte ich das geschrieben." + instruction_Grammer

    thread1 = threading.Thread(target=setup_server, args=(5002, instruction_Grammer))
    thread2 = threading.Thread(target=setup_server, args=(5004, instruction_Letter))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

if __name__ == "__main__":
    main()
