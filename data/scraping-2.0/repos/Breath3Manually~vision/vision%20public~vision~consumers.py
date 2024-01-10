import json
from channels.generic.websocket import AsyncWebsocketConsumer
import openai
import asyncio
from visionApp.models import Message, ChatbotConfiguration, Character, Conversation
from channels.db import database_sync_to_async
from google.cloud import texttospeech
import os

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.character = None
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def send_complete_signal(self):
        await self.send(text_data=json.dumps({'message_complete': True}))

    @database_sync_to_async
    def save_user_message(self, message_text, conversation):
        user_message = Message(text=message_text, is_user=True, conversation=conversation)
        user_message.save()
        return user_message
        
    @database_sync_to_async
    def save_ai_message(self, message_text, conversation):
        ai_message = Message(text=message_text, is_user=False, conversation=conversation)
        ai_message.save()
        return ai_message

    @database_sync_to_async
    def get_or_create_character(self, character_name):
        character = None
        try:
            character, created = Character.objects.get_or_create(character_name=character_name)
        except Exception as e:
            print(f"Error in get_or_create: {e}")
        return character

    @database_sync_to_async
    def get_or_create_conversation(self, character):
        conversation, created = Conversation.objects.get_or_create(character=character)
        return conversation

    @database_sync_to_async
    def get_character_messages(self, character):
        conversation = Conversation.objects.get(character=character)
        return list(conversation.messages.all())
    
    @staticmethod
    async def text_to_speech(text, base_path='media/audio_messages/output'):
        file_number = 1
        output_file = f'{base_path}{file_number}.mp3'
        while os.path.exists(output_file):
            file_number += 1
            output_file = f'{base_path}{file_number}.mp3'
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code='en-US',
            name='en-US-Neural2-F',
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.3
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(output_file, "wb") as out:
            out.write(response.audio_content)

            return f'/media/audio_messages/output{file_number}.mp3'

    @database_sync_to_async
    def get_character_by_name(self, character_name):
        try:
            name = Character.objects.get(character_name=character_name)
            return name
        except Character.DoesNotExist:
            return None

    async def receive(self, text_data):
        data = json.loads(text_data)
        character_name = None
        
        # Check if the incoming message is a command to change the chatbot's personality.
        if 'command' in data and data['command'] == 'change_personality':
                character_name = data.get('character_name')
                details = data.get('details')
                scenario = data.get('scenario')
                personality = data.get('personality')
                examples = data.get('examples')
                initial_message = data.get ('initial_message')
                
                if character_name:
                    # Parse the new personality JSON string.
                    self.character = await self.get_or_create_character(character_name)

                    self.character.details = "character details: " + (details or '')
                    self.character.name = "this is your name: " + (character_name or '')
                    self.character.personality = "this is your personality: " + (personality or '')
                    self.character.scenario = "this is the scenario you will be roleplaying: " + (scenario or '')
                    self.character.examples = "Here are some example dialogues: " + (examples or '')

                    # Call the method that updates the chatbot's personality in the database.
                    if self.character is not None:
                        await database_sync_to_async(self.character.save)()
                    else:
                        print("self.character is none")

                #send initial message if no messages in conversation
                conversation = await self.get_or_create_conversation(self.character)
                character_messages = await self.get_character_messages(self.character)
                if not character_messages:
                    await self.save_ai_message(initial_message, conversation)
        else:
            message_text = data.get('message')
            character_name = data.get('character_name')

            if message_text:
                if self.character is None or self.character.character_name != character_name:
                    self.character = await self.get_character_by_name(character_name)
                    
                if self.character is not None:
                    conversation = await self.get_or_create_conversation(self.character)
                    # Save the user's message to the database.
                    await self.save_user_message(message_text, conversation)

                    # Construct the messages list for OpenAI input.
                    messages = [{"role": "system", "content": self.character.personality}]

                    # Add all previous messages to the list.
                    character_messages = await self.get_character_messages(self.character)
                    for message in character_messages:
                        role = "user" if message.is_user else "assistant"
                        messages.append({"role": role, "content": message.text})

                    character_prompt = (
                    f"Name: {self.character.character_name}\n"
                    f"Details: {self.character.details}\n"
                    f"Personality: {self.character.personality}\n"
                    f"Scenario: {self.character.scenario}\n"
                    f"Examples: {self.character.examples}"
                    )       

                    # Add the current personality as a system message.
                    messages.append({"role": "system", "content": character_prompt})

                    # Call OpenAI's API here with the 'messages' structure.
                    openai.api_key = 'sk-'
                    response = openai.ChatCompletion.create(
                        #model='gpt-3.5-turbo',
                        model='gpt-4-1106-preview',
                        messages=messages,
                        temperature=0.6,
                        max_tokens=2000,
                        stream=True
                    )

                    # Process the response from OpenAI to generate the chatbot's reply.
                    collected_messages = []
                    for message_chunk in response:
                        chunk_message = message_chunk['choices'][0]['delta']
                        if 'content' in chunk_message:
                            collected_messages.append(chunk_message['content'])
                            await self.send(text_data=json.dumps({'message_chunk': chunk_message['content']}))
                            await asyncio.sleep(0.0001)  # This delay may need adjustment.

                    # Combine the chunks to form the full message.
                    chatbot_message = ''.join(collected_messages).strip()

                    # Convert AI's message to speech
                    audio_path = await self.text_to_speech(chatbot_message)
                    
                    # Send the message and audio file path to the frontend
                    await self.send(text_data=json.dumps({
                        'message': chatbot_message,
                        'audio_path': audio_path
                    }))

                    # Save the AI's message to the database.
                    await self.save_ai_message(chatbot_message, conversation)

                    # Send the full message to the frontend.
                    await self.send_complete_signal()
                else:
                    print('dingle oh no...')