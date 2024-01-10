import openai
from PIL import Image as Img
from urllib.request import urlopen
import os
import random
import string
from abc import ABC, abstractmethod

class AICommand(ABC):
    @abstractmethod
    def processing(self):
        pass

class Image(AICommand):
    def processing(self):
        description = input('Enter a description of image you want create: ')
        if not os.path.exists('created_images'):
            os.mkdir('created_images')
        url = openai.Image.create(
        prompt=description,
        n=2,
        size="1024x1024"
        )['data'][0]['url']
        image = Img.open(urlopen(url))
        image_path = os.path.join('created_images', 'image_' + ''.join(random.choices(string.ascii_letters, k = 10)) + '.png' )
        image.save(image_path)
        image.show()
        return f'\033[032mYour image has been created and saved to \033[0m{image_path} '

class Question(AICommand):
    def processing(self):
        while True:
            user_question = input('Enter your question (type exit to stop): ')
            if user_question == 'exit':
                break
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_question}])
            print(completion.choices[0].message['content'])
        return '\033[032m Thank you for using our AI helper\033[0m'
    
COMMANDS = {
    'generate image': Image,
    'ask question' : Question,
}

def get_handler(action):
    for command in COMMANDS.keys():
        if action == command:
            return COMMANDS[command]()


#def performer(command: AICommand):
 #       return command.processing()

