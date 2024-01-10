from openai import OpenAI
from threading import Thread, Event
from queue import Queue
import argparse
from agent import GPTVisionAgent
from image_handler import ImageHandler
from time import sleep
from cv2 import waitKey


system_prompt = (
    "Você se chama Clio e é uma Inteligência Computacional Autônoma (ICA) "
    "do laboratório de Computação de Alto Desempenho (LCAD) da Universidade "
    "Federal do Espírito Santo (UFES).\n"
    "Você é uma barista muito prestativa e é responsável por instruir o processo de fazer café coado da forma "
    "mais detalhada possível e em qualquer configuração de cozinha residencial em que esteja. Deverá me guiar "
    "fornecendo instruções sequenciais para o preparo do café, considere que será usado café em pó.\n"
    "Você deve ser capaz de guiar um usuário que nunca preparou café antes,"
    "sempre pergunte se o usuário tem o item necessário para a tarefa e se o item é próprio para a tarefa, "
    "só prossiga com a tarefa se o usuário confirmar que tem o item.\n"
    "Suas instruções serão claras e diretas, não mais do que uma tarefa por vez e limite de 100 caracteres por tarefa. "
    "Exemplos de interações:\n"
    "(EXEMPLO)'user': 'Clio, me pergunte se podemos iniciar'; 'system': 'Podemos iniciar o preparo do café?'; 'user': 'Sim';\n"
    "(EXEMPLO)'system': 'Verifique se você tem um recipiente para ferver a água\n"
    "(EXEMPLO)'user': 'Passo concluído.'; 'system': 'Encontre uma torneira'\n"
    "(EXEMPLO)'user': 'Passo concluído.'; 'system': 'Coloque água no recipiente'\n"
)


def show_webcam():
    camera_handler = ImageHandler(ip_address)
    while True:
        if capture_frame_event.is_set():
            capture_frame_event.clear()
            queue.put(camera_handler.capture_webcam_frame())
            frame_captured_event.set()
        camera_handler.show_webcam()
        

def main():
    client = OpenAI()
    coffee_assistant = GPTVisionAgent(system_prompt=system_prompt,
                                      model="gpt-4-vision-preview",
                                      image_history_rule='none')
    while True:
        user_response = input("User: ")
        for i in range(3):
            sleep(0.75)
            print(f"Capturing frame in {3 - i}...")
        capture_frame_event.set()
        frame_captured_event.wait()
        image = queue.get()
        response = coffee_assistant.get_response(client, image, user_response)
        print("Assistant: " + response)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A program that does something.")
    parser.add_argument('ip_address', type=str,
                        help='The camera IP address to use')
    args = parser.parse_args()
    ip_address = args.ip_address

    capture_frame_event = Event()
    frame_captured_event = Event()
    queue = Queue()

    thread_show_webcam = Thread(target=show_webcam)
    thread_show_webcam.start()

    main()

    thread_show_webcam.join()
