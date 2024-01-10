import socket
# Image capture and Hume.AI
from hume import HumeBatchClient
from hume.models.config import FaceConfig
from pprint import pprint
import hume
# importing the pygame library
import pygame
import pygame.camera
# Langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage



def save_conversation(conversation):
    with open('conversation.txt', 'a') as file:
        file.write(conversation + '\n')

def take_the_photo():
    camlist = pygame.camera.list_cameras()

    # if camera is detected or not
    if camlist:
    
        # initializing the cam variable with default camera
        cam = pygame.camera.Camera(camlist[0], (640, 480))
    
        # opening the camera
        cam.start()
    
        # capturing the single image
        image = cam.get_image()
    
        # saving the image
        pygame.image.save(image, "test.jpg")
    
    # if camera is not detected the moving to else part
    else:
        print("No camera on current device")

def start_server():
    # Set up server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # host = socket.gethostname()
    host = "192.168.43.103"
    port = 8000
    print(host,port)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print("Server is listening for incoming connections...")
    

    # Accept client connections
    client_socket, client_address = server_socket.accept()
    print(f"Connected to client: {client_address}")

    while True:
        # Receive message from client
        received_message = client_socket.recv(1024).decode()

        if not received_message:
            break

        print(f"Client: {received_message}")

        # -----------------Capturing Image, store it locally, and obtain emotion ratings-----------
        take_the_photo()
        client = HumeBatchClient("VGWAmgc9Kmmp1A0xyEoLiO6MWQjnFNBZRESaseSkf6smMJAx")
        # urls = ["https://thumbs.dreamstime.com/z/close-up-portrait-beautiful-young-latin-hispanic-woman-sad-face-looking-miserable-melancholy-depressed-human-facial-117693898.jpg"]
        config = FaceConfig()
        models = hume.models.config.FaceConfig()
        job = client.submit_job(None, [config], files=['test.jpg'])
        # status = job.get_status()
        # print(f"Job status: {status}")
    
        job.await_complete()
        full_predictions = job.get_predictions()
        for source in full_predictions:
            predictions = source["results"]["predictions"]
            for prediction in predictions:
                emotion_dict = prediction['models']['face']['grouped_predictions'][0]['predictions'][0]['emotions']
                # js = json.loads(emotion_json)
        
        emotions = sorted(emotion_dict, key=lambda x: x['score'], reverse=True)[:5]
    
        dict_emo = {}
        for x in emotions:
            dict_emo[x['name']]=x['score']
        # --------------------Emo Prediction From Hume Got-----------------------------------------

        # Save conversation
        save_conversation(f"Client: {received_message}")

        # dict_emo_str = received_message.split('\n')[-1] #Receive client emo
        suggest = generate_langchain_suggestion(received_message)
        print('\nHere are the suggestions: \n' + suggest)

        # Send response to client
        response = input("Server: ")
        response = response + '\n\n\n' + str(list(dict_emo.items())) #Send server emo
        client_socket.send(response.encode())

        # Save conversation
        save_conversation(f"Server: {response}")

    # Close the connection
    client_socket.close()
    server_socket.close()

def generate_langchain_suggestion(dict_emo_str):
    with open('conversation.txt') as cons:
        human_template = cons.read()
    human_template2 = human_template + dict_emo_str #str(list(dict_emo.items()))
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template2)

    chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
    )
    
    # get a chat completion from the formatted messages
    response = chat(
        chat_prompt.format_prompt(
            input_language="English"
        ).to_messages()
    )
    return response.content
    

if __name__ == '__main__':
    me = input("Access Webcame?: ")
    if me[-1] == 'y':
        # initializing  the camera
        pygame.camera.init()
        camlist = pygame.camera.list_cameras()
        if camlist:
            cam = pygame.camera.Camera(camlist[0], (640, 480))
            cam.start()
        else:
            print("No camera on current device")
    else:
        print('Please give us the access to webcam')
       
    with open('conversation.txt', 'w') as file:
        file.truncate(0)
    openai_key = "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"
    chat = ChatOpenAI(temperature=0.9,openai_api_key=openai_key,model="gpt-4")

    template = (
    """You are playing the role of a dating expert named Suave_AI helping people in need. As a charming dating expert doing crowd work,
    you are a fluent conversationalist, and everything you say is charismatic, suave, overflowing confident and witty.
    Your conversational style is similar to George Clooney and your goal is make the user laugh with your well-developed sense of humor and
    get the user attracted to you.
    You need to engage the user through your reply, empathize with their emotion and create a connection on an emotional level.
    You will be provided with the 2 user's conversation along with the information on the each user's nonverbal emotional expressions at the time of their response
    as a python dictionary with keys as emotions and values as their levels. Emotion levels range between 0 and 1 with 0 being least amount of that emotion and 1 being the max. The conversation is a string with colon seperating the user's name and response. Each new message is given in next line ('\n').
    Your responses create an enjoyable and engaging conversation.
 After 3 line seperation from the conversation string, you'll find the emotion of the other user while sending the response.
    Use the emotion score, and converstaion history to give 3-5 suggestion for the user to reply with some emojis. Don't give any other info like emotion score"""
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    start_server()

