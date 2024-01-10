import pygame 
from settings import *
from support import import_folder
import random
from dialog import Dialog

import openai
import requests
import simpleaudio as sa
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
import threading
from LLM import set_LLM_Mood,answer_with_mood_LLM,function_call_create_item_from_NPC
url = "https://app.coqui.ai/api/v2/samples"



class NPC(pygame.sprite.Sprite):
	def __init__(self,pos,groups,obstacle_sprites, name):
		super().__init__(groups)
		self.name = name
		self.image =  pygame.image.load(f'graphics/npc/{self.name}/idle/tile0000.png').convert_alpha()
		self.rect = self.image.get_rect(topleft=pos)
		self.hitbox = self.image.get_rect(topleft=pos) 
		self.frame_index = 0
		self.animation_speed = 300
		self.last_update = pygame.time.get_ticks()  # time when the frame was last updated
		self.is_interacting = False
		self.font = pygame.font.Font(None, 20)  
		self.player=None
		self.isinteracting=False
		self.npc_story=NPC_stories[self.name]
		self.voice=NPC_VOICES[self.name]
		self.sys_msg={"role": "system", "content": f'You are to speak like an NPC in a rogue-like game. The main story of the game is the following: {STORY}. Inside that story, you are {self.npc_story}.'}
		self.messages=[self.sys_msg]
		#self.agent= OpenAiAgent(model="gpt-4", api_key=API_KEY)


		self.dialog =Dialog(self, "", self.font) 
		# graphics setup
		self.import_npc_assets()
		self.status = 'idle'
	def end_interaction(self):
		self.isinteracting=False
		print("ending interaction with:",self.player)
		self.player.can_move=True
		pass
	def interact(self,player,event=None):
		# Now you can access the name of the NPC in your interactions
		self.player=player
		player.can_move=False
		self.isinteracting=True
		
		print(f"Interaction with {self.name}!")
		print(self.player.can_move)
		self.dialog.player=player
		self.dialog.is_visible = True # Dialog becomes visible when interaction happens
  
	def get_npc_response(self, player_input):
	# This function represents the NPC's "brain". It takes the player's input and
	# decides what the NPC should say in response. This is a simple example, but you
	# could make this as complex as you want.
		#out=self.agent.chat(player_input)
		self.messages.append({"role": "user", "content": player_input})
		
		mood=set_LLM_Mood(player_input)
  
		assistant_msg=answer_with_mood_LLM(player_input,self.messages,mood)
  
		self.messages.append({"role": "assistant", "content": assistant_msg})
		payload = {
			"emotion": f"{mood}",
			"speed": 1,
			"voice_id": f"{self.voice}",
			"text": f"{assistant_msg}"
		}
		headers = {
			"accept": "application/json",
			"content-type": "application/json",
			"authorization": "Bearer <Coqui-Key>"
		}
		try:
			response = requests.post(url, json=payload, headers=headers)

			audio_url = response.json()["audio_url"]

				# Download audio file
			audio_data = requests.get(audio_url).content

			# Load audio to pydub
			audio = AudioSegment.from_file(BytesIO(audio_data), format="wav")

			# Play audio in a separate thread
			threading.Thread(target=play, args=(audio,)).start()
		except:
			print("oups")

		return assistant_msg,mood
	def get_status(self):
		# Define the possible statuses and their probabilities
		statuses = ['idle', 'action']
		probabilities = [0.8, 0.2]  # 80% for 'idle', 20% for 'action'

		# Set the status
		self.status = random.choices(statuses, probabilities)[0]
	
	def animate(self):
		now = pygame.time.get_ticks()
		if now - self.last_update > self.animation_speed:
			self.last_update = now
			self.frame_index = (self.frame_index + 1) % len(self.animations[self.status])

			# Check if we've looped back to the start of the animation
			if self.frame_index == 0:
				self.get_status()

			self.image = self.animations[self.status][int(self.frame_index)]
			self.rect = self.image.get_rect(center = self.hitbox.center)
   


	def import_npc_assets(self):
		character_path = f'graphics/npc/{self.name}/'
		self.animations = {'idle': [],'action': [],
		}

		for animation in self.animations.keys():
			full_path = character_path + animation
			self.animations[animation] = import_folder(full_path)

	def update(self):
		if self.is_interacting:
			self.interact()

		self.animate()

		if self.dialog.is_visible:
      
      
			self.dialog.draw()
			self.dialog.update()
