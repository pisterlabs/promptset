import os
import openai
import pandas as pd

openai.api_key = ""

data = pd.read_csv('testData.csv')
data.dropna()
del data["Domain"]

data['Comptetencies'] = data['Competency Code'] + " " + data['Competency Description']

data['Micro-competencies'] = data['Micro-competency'] + " " + data['Micro-competency Description']
data['Micro-competencies'].dropna(inplace=True)

dictionary = data.applymap(str).groupby('Comptetencies')['Micro-competencies'].apply(list).to_dict()

i = 0
game_names = "games = [overworld_game()"
imports = ""
#for key in dictionary:
key = next(iter(dictionary))
game_filename = "game"+str(i)

#game generation
msg1 = """Your role is to generate code in educational games to teach children. Your code must implement the 'iter_game' class and use pygame functions to display elements on the screen. Refrain from including code explanations and only implement the iter_game class as a new subclass.
"You will create educational minigames based off the learning goals. The minigame should have a library setting. The learning goals are what the student should learn by the end of the minigame. Each learning goal has a minigame. Each learning goal will have sub-goals. The student must have learned every sub-goal associated with it's learning goal to be considered having learned the learning goal. Each sub-goal must be covered in a minigame. The minigame for each learning goal must include each subgoal. Each minigame should have the learning goal code, a description, game mechanics, a game flow that goes through what the player does, and the sub-goals."
The following is the class 'iter_game':
###
import pygame

class iter_game(object):
	def __init__(self,window_width=50,window_height=50):
		self.fin = False
		self.window_height = window_height
		self.window_width = window_width
	def next(self,pygame,events,screen,dt,data):
		###Events contains the events returned from pygame.event.get()
		###Screen is the pygame's display
		###dt is the time elapsed
		###data is an object to store information between games
		if self.fin:
			data.game_num = 0
			return
		###Write code here

		###
		pygame.display.flip()
###

'iter_game' is used in the context of this game loop:
###
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False
    games[game_num].next(pygame,events,screen,dt,dobj)
    game_num = dobj.game_num
    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000
###
"""
#removed: Text: Create a game to play snake by implementing the 'iter_game' class.
msg2 = """"The learning goal is: Direct basic moving shapes \n
+ "The sub-goals are: \n\n Understand different colors, Evaluate the speed of moving objects, Locate position of important objectives\n
Result:
###
from iter_game import iter_game
import pygame
import random
# Initialize the game
# Set up the game window
window_width = 800
window_height = 600
# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
# Define game variables
snake_size = 20
snake_speed = 10
clock = pygame.time.Clock()
# Define the Snake class
class Snake:
	def __init__(self):
		self.x = window_width // 2
		self.y = window_height // 2
		self.direction = "RIGHT"
		self.length = 1
		self.body = []
	def move(self):
		if self.direction == "UP":
			self.y -= snake_size
		elif self.direction == "DOWN":
			self.y += snake_size
		elif self.direction == "LEFT":
			self.x -= snake_size
		elif self.direction == "RIGHT":
			self.x += snake_size
	def draw(self,pygame,screen):
		for part in self.body:
			pygame.draw.rect(screen, GREEN, (part[0], part[1], snake_size, snake_size))
	def check_collision(self):
		if self.x < 0 or self.x >= window_width or self.y < 0 or self.y >= window_height:
			return True
		for part in self.body[1:]:
			if self.x == part[0] and self.y == part[1]:
				return True
		return False
# Define the Food class
class Food:
	def __init__(self):
		self.x = random.randint(0, (window_width - snake_size) // snake_size) * snake_size
		self.y = random.randint(0, (window_height - snake_size) // snake_size) * snake_size
	def draw(self,pygame,screen):
		pygame.draw.rect(screen, RED, (self.x, self.y, snake_size, snake_size))
# Initialize the snake and food
snake = Snake()
food = Food()

class snake_game(iter_game):
	def __init__(self,window_width=50,window_height=50):
		self.i = 2
		self.window_height = window_height
		self.window_width = window_width
		self.fin = False
	def next(self,pygame,events,screen,dt,data):
		global snake
		global food
		if self.fin:
			data.game_num = 0
			return
		for event in events:
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_UP and snake.direction != "DOWN":
					snake.direction = "UP"
				elif event.key == pygame.K_DOWN and snake.direction != "UP":
					snake.direction = "DOWN"
				elif event.key == pygame.K_LEFT and snake.direction != "RIGHT":
					snake.direction = "LEFT"
				elif event.key == pygame.K_RIGHT and snake.direction != "LEFT":
					snake.direction = "RIGHT"
		# Move the snake
		snake.move()
		# Check collision with food
		if snake.x == food.x and snake.y == food.y:
			snake.length += 1
			food = Food()
		# Update the snake's body
		snake.body.insert(0, (snake.x, snake.y))
		if len(snake.body) > snake.length:
			snake.body.pop()
		# Check collision with snake's body or boundaries
		if snake.check_collision():
			data.game_num = 0
			self.fin = True
		# Clear the window
		screen.fill(BLACK)
		# Draw the snake and food
		snake.draw(pygame,screen)
		food.draw(pygame,screen)
		# Update the display
		pygame.display.update()
		# Set the game speed
		clock.tick(snake_speed)

###
"""
subGoals = "The sub-goals are: \n" + "\n".join(dictionary[key])
msg3 = "The learning goal is: \n" + key + "\n" + subGoals + """
from iter_game import iter_game
"""
msgs = [{
	 "role": "system","content":msg1
},
{
	 "role": "system","content":msg2
},
{
	 "role": "user","content":msg3
}]
completion = openai.ChatCompletion.create(model='gpt-4',messages=msgs,temperature=0.2)
file = open(game_filename+".py","w")
file.write("from iter_game import iter_game\n" + completion.choices[0].message.content)
msgsfu = msgs + [{
	 "role": "system","content":completion.choices[0].message.content
},
{
	 "role": "user","content":"""Output the name of the class for the previous generated file
	 Text: class animal_game(iter_game):
	 Result: animal_game

	 Text: class snake_game(iter_game):
	 Result: snake_game

	 Text: """ + completion.choices[0].message.content + """
	 Result:"""
}]
completion = openai.ChatCompletion.create(model='gpt-4',messages=msgsfu,temperature=0.2)
print(completion.choices[0].message.content)

imports += "".join(["from ", game_filename, " import ",completion.choices[0].message.content, "\n"])
game_names += "".join([",",completion.choices[0].message.content,"()"])

with open("overworld.py", "r") as f:
    contents = f.readlines()
game_names+="]"
contents.insert(70,game_names)
contents.insert(8,imports)
with open("run.py", "w") as f:
    contents = "".join(contents)
    f.write(contents)

