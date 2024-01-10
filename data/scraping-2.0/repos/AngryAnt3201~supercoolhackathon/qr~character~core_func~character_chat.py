#Give quest 
#Complete quests
#Interact and chat



#Game Context 
#user context -> Current quests, other characters 

import openai
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')

class Character_AI: 

    def __init__(self, name, description, game_context, quests):
        self.description = description
        self.name = name
        self.messages = [
            {'role': 'system', 'content': 'You are a virtual character within a video game, roleplay according to your character description: ' + str(self.description)},
            {'role': 'system', 'content': 'This is what has happened in the game so far: ' + str(game_context)}, 
            {'role': 'system', 'content': 'These are the quests the player has been given that involve you: ' + str(quests)}
        ]
    #Get GPT to check for quest completion, etc. 
    def complete_quest(self, message_segment):
        pass

    #Call create quest endpoint 
    def issue_quest(self, message_segment):
        pass

    #Pigmalion Hopefully, automatically route to generate dialog
    def generate_dialogue(self, user_input):

        self.messages.append({'role': 'user', 'content': user_input})
        response = openai.ChatCompletion.create(
            model='gpt-4-0613',
            temperature=0.1,
            messages = self.messages, 
            max_tokens=200,
        )
        response_message = response["choices"][0]["message"]

        try:
            function_name = response_message["function_call"]["name"]
            return response['choices'][0].message['content'], True

        except KeyError:
            print("Nothing to confess")

        self.messages.append({'role': 'system', 'content': response['choices'][0].message['content']})
        return response['choices'][0].message['content']








        