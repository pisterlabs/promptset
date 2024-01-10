import socket
import openai

openai.api_key = "" #Poner tu API de openai


class Server:

    def __init__(self, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('localhost', port))
        self.socket.listen(1)
        self.context = None

    def run(self):
        while True:

            clientSocket, address = self.socket.accept()
            print('Received connection from', address)

            while True:
                message = clientSocket.recv(1024)
                if message == 'exit':
                    break
                print('Received message:', message)

                message_string = message.decode()
                
                if self.context:
                    # Si hay contexto, agregarlo al mensaje del usuario
                    message_string = self.context + " " + message_string
                #Aqui poner el rol del asistente
                messages = [{"role": "system", "content": ""},
                            {"role": "user", "content": message_string}]

                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.2
                )

                self.context = response.choices[0].message.content
                
                clientSocket.sendall(response.choices[0].message.content.encode())

            clientSocket.close()

if __name__ == '__main__':
    server = Server(8080) #Puerto por defecto 8080
    server.run()
