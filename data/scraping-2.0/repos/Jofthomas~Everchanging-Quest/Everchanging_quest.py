import pygame, sys
from settings import *
from level import Level

from os import walk
import openai


def import_folder(path):
    surface_list = []

    for _,__,img_files in walk(path):
        for image in img_files:
            full_path = path + '/' + image
            image_surf = pygame.image.load(full_path).convert_alpha()

            # Get the original image size
            original_size = image_surf.get_size()

            # Calculate the new size with scaling factor of 1.3
            new_size = (int(original_size[0]*3), int(original_size[1]*3))

            # Scale the image
            image_surf = pygame.transform.scale(image_surf, new_size)

            surface_list.append(image_surf)

    return surface_list

class Menu:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 50)  # Change to desired font
        self.base_image = pygame.image.load('graphics/UI/Menu/Menu.png')
        self.title_image = pygame.image.load('graphics/UI/Menu/Title.png')
        self.play_image = pygame.image.load('graphics/UI/Menu/Play col_Button.png')
        self.pause_image = pygame.image.load('graphics/UI/Menu/Exit  col_Button.png')
        self.animation_images = import_folder('graphics/UI/Menu/Portal')

        self.original_play_size = self.play_image.get_size()
        self.original_pause_size = self.pause_image.get_size()

        self.new_play_size = (int(self.original_play_size[0]/4), int(self.original_play_size[1]/4))
        self.new_pause_size = (int(self.original_pause_size[0]/4), int(self.original_pause_size[1]/4))

        self.play_image = pygame.transform.scale(self.play_image, self.new_play_size)
        self.pause_image = pygame.transform.scale(self.pause_image, self.new_pause_size)

        self.current_frame = 0
        self.title_x, self.title_y = 100, 50

        self.play_button_rect = self.play_image.get_rect(center=(WIDTH / 2, HEIGTH / 2))
        self.pause_button_rect = self.pause_image.get_rect(center=(WIDTH / 2, HEIGTH / 2 + self.play_button_rect.height + 30))
        self.frame_index = 0.0
        self.animation_speed = 0.4

    def draw(self):
        self.screen.blit(self.base_image, (0, 0))
        self.screen.blit(self.animation_images[int(self.frame_index)], (20, -10))
        self.screen.blit(self.title_image, (self.title_x, self.title_y))
        self.screen.blit(self.play_image, self.play_button_rect)
        self.screen.blit(self.pause_image, self.pause_button_rect)

        self.frame_index += self.animation_speed
        if self.frame_index >= len(self.animation_images):
            self.frame_index = 0

        pygame.display.update()
        

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.play_button_rect.collidepoint(event.pos):
                return 'game'
			
            elif self.pause_button_rect.collidepoint(event.pos):
                pygame.quit()
                sys.exit()
				
        return 'menu'

class Game:
	def __init__(self):
		  
		# Initialize your OpenAI API key
		openai.api_key = API_KEY
		# general setup
		pygame.init()
		self.screen = pygame.display.set_mode((WIDTH,HEIGTH))
		pygame.display.set_caption('Everchanging quest')
		self.clock = pygame.time.Clock()
		self.main_sound = pygame.mixer.Sound('audio/xDeviruchi - The Final of The Fantasy.wav')
		self.main_sound.set_volume(0.1)
		self.level = Level(self.main_sound)
		self.menu = Menu(self.screen)
		self.state = 'menu'
		
		
		self.menu_sound = pygame.mixer.Sound('audio/xDeviruchi - Title Theme .wav')
		self.menu_sound.set_volume(0.05)		

		
	
	def run(self):
		while True:
			for event in pygame.event.get():
				self.event=event
				if event.type == pygame.QUIT:
					pygame.quit()
					sys.exit()
				elif self.state == 'menu':
					self.menu_sound.play(loops = -1) 
					new_state = self.menu.handle_event(event)
					if new_state == 'game':
						self.menu_sound.stop()
						self.main_sound.play(loops = -1)  # start playing music
					self.state = new_state
				elif self.state == 'game':
					if event.type == pygame.KEYDOWN:
						if event.key == pygame.K_ESCAPE:
							self.main_sound.stop()  # stop playing music
							self.state = 'menu'
						if event.key == pygame.K_m:
							self.level.toggle_menu()

			if self.state == 'menu':
				self.menu.draw()
				
			elif self.state == 'game':
				self.screen.fill((0,0,0))
				self.level.run(self.event)
				self.menu_sound.stop()
			
			pygame.display.update()
			self.clock.tick(FPS)

if __name__ == '__main__':
	game = Game()
	game.run()