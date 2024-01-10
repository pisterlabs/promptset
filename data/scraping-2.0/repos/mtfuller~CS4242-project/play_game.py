"""A simple script to allow a human to play the CartPole game.

This python script is used to allow a human player to experience the CartPole
gym environment, and get a feel for what the agent will need to learn and
understand.
"""

# Import OpenAI's gym library, used to intialize a CartPole agent environment
import gym

# Import pygame to allow a human to play the agent environment
import pygame

# Numpy used for simple image matrix manipulation (rotating image matrix) and
# mean calculation
import numpy as np

# Define the FPS for the game, lower the easier to play
FPS = 10

# Creates a new gym CartPole environment
env = gym.make('CartPole-v1')
observation = env.reset()
firstframe = env.render(mode='rgb_array')

# Starts pygame
pygame.init()

# Sets up font for text
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 30)

# Set the dimensions of the game to the same as the first frame of the CartPole
# environment
display_height, display_width, chan = firstframe.shape

# Initialize the pygame GUI window
gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Human Playable Game')

# Define colors for black and white
black = (0,0,0)
white = (255,255,255)

# Setup main pygame clock
clock = pygame.time.Clock()

# Boolean that is used to tell if pygame should quit
crashed = False

def render_frame():
    """Renders the current frame/state of the CartPole environment

    Retrieves the current state from the Gym CartPole environment in the form of
    an RGB color matrix. This matrix is then rendered to the pygame window.
    """
    frame = np.flip(np.rot90(env.render(mode = 'rgb_array')), axis=0)
    gameDisplay.blit(pygame.surfarray.make_surface(frame), (0,0))

def render_text(text, coor):

    """Renders a message to the pygame screen

    Renders the given string to the pygame window at the specified coordinates.

    Args:
        text: A string of the text to be rendered to the screen.
        coor: A tuple representing the x-y coordinate pair in the form of (x,y).
    """
    gameDisplay.blit(myfont.render(text, False, (0, 0, 0)), coor)

# Defines the initial action to be taken. Accelerate left is '1', while
# accelerating right is '0'
action = 0

# Defines the initial value of the score
score = 0

# Initializes a list of scores that the player had earned
scores = []

# Continues until the game window is closed
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # If close button is hit, exit the game
            crashed = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                # Begins to accelerate the cart to the left
                action = 0
            if event.key == pygame.K_RIGHT:
                # Begins to accelerate the cart to the right
                action = 1

    # Perform the current action in the environment
    observation, reward, done, info = env.step(action)

    # Increase the player's score for the current game
    score += reward

    # If the current gym environment game has been won/lost, let's reset the
    # environment and the player's score.
    if done:
        scores.append(score)
        print("GAME",len(scores),"FINISHED! Score:",score)
        score = 0
        env.reset()

    # Draw the game scene
    gameDisplay.fill(white)
    render_frame()
    render_text("Score: "+str(int(score)), (5,5))

    # Update the pygame window
    pygame.display.update()
    clock.tick(FPS)

# Display average human score
print("Average Score:",np.mean(scores))

# Quit the game
pygame.quit()
quit()
