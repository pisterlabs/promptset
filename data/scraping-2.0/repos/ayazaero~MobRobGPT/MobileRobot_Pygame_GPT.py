import pygame
import sys
import numpy as  np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
from robot import Robot #kinematics of mobile robot, will be included in coming iterations
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d
import random
import warnings
from openai import OpenAI
import os
import time

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

# Filter DeprecationWarning
warnings.filterwarnings("ignore")

#_______________________________________________________________________________________________________________________________________________#

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
TEAL = (0,128,128)

# Define circle and obstacle properties
circle_radius = 10
circle_positions = [(450, 50), (50, 250), (600, 500), (100, 500), (650, 275)]
circle_colors = [BLUE, RED, RED, GREEN, GREEN]
new_circle_positions = [(x, 800 - y) for x, y in circle_positions]
c_colors=['BLUE', 'RED', 'RED', 'GREEN', 'GREEN']
obstacle_positions = [(150, 500, 350, 50), (100, 250, 500, 50), (700, 100, 50, 500),(0,0,800,5),(0,795,800,5),(0,5,5,790),(795,5,5,790)]

circle_info = [f"{color}:({x},{y})" for (x, y), color in zip(new_circle_positions, c_colors)]
obstacle_info = [f"({x},{y},{w},{h})" for x, y, w, h in obstacle_positions]

circle_text = '; '.join(circle_info)
obstacle_text = ', '.join(obstacle_info)


# Create text for GPT-3.5 instructions
text_loc = f"Locations of dots are [{circle_text}] and obstacles are given as [{obstacle_text}]"

text_pre="""There is a mobile robot, you are doing path planning for the robot using artificial potential functions. the space is 800X800 square with 5 unit thick boundary on each side. there are some rectangular obstacles, the location for which is given in the format (a,b,c,d) where a,b is the location of the top right corner in x,y plane and c,d is the widht and height of the obstacle. the number of obstacles can be determined based oin the data.
    there are also some points represented as circles, for each point you will be given a location and color of the circle.
    Your aim is to decide the goal based on the prompt. in some cases you will be asked to avoid certain region, in that case you put a repulsor which will repel the robot away from that point. In the end robot will move towards the goal and avoid repulsors in case they are specified."""

text_rules="""It is absolutely essential that you always follow these rules:
    1. Give answers "strictly", Strictly in this format $goal:(x_goal,y_goal); avoid:[(x1,y1);(x2,y2)]$
    2. If there are no repulsors, give answer in this format $goal:(x_goal,y_goal)$
    3. Do not give any response which is not tin the format mentioned in rule 1
    4. There can be only one goal but multiple repulsors.
    5. Use repulsors only when asked to avoid certain area.
    The input will be given in the form of general instructions like "go to red circle", you select the goal and repulsor based on the instructions"""

guidelines = text_pre + "\n\n" + text_loc + "\n\n" + text_rules #guidelines for GPT-3.5

#  GPT-3.5 related setup
my_assistant = client.beta.assistants.create(
    instructions=guidelines,
    name="MobRob",
    tools=[{"type": "code_interpreter"}],
    #model="gpt-4",
    model="gpt-3.5-turbo-1106",
)

thread = client.beta.threads.create()

def create_message(thread_id, current_location, user_input):
    # Convert current location list to a string
    location_str = f"{current_location[0]}, {current_location[1]}"

    # Create the content for the message
    content = f"The current location is {location_str}, {user_input}"

    # Assuming you have a 'client' object available, use it to create the message
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )

def draw_robot(x, y, angle_rad):    #for drawing robot agent location

    # Calculate vertices of the triangle based on robot position and orientation
    point1 = (int(x+20*np.cos(angle_rad)), int(y+20*np.sin(angle_rad)))
    point2 = (
        int(x - 5 * np.cos(angle_rad) + 10 * np.sin(angle_rad)), 
        int(y - 5 * np.sin(angle_rad) - 10 * np.cos(angle_rad))
    )
    point3 = (
        int(x - 5 * np.cos(angle_rad) - 10 * np.sin(angle_rad)), 
        int(y - 5 * np.sin(angle_rad) + 10 * np.cos(angle_rad))
    )

    pygame.draw.polygon(screen, TEAL, (point1, point2, point3))


########################################################################################################################################################
# For potential field
# Define workspace boundaries

corner=[[5,5],  #corner to activate in case, agent is stuck in local minima
        [5,795],
        [795,795],
        [795,5]];

x_min, y_min = 0, 0  # boundaries
x_max, y_max = 800.0, 800.0  

obstacle=obstacle_positions
# Define grid parameters
resolution = 5  
x_grid = np.arange(x_min, x_max, resolution)
y_grid = np.arange(y_min, y_max, resolution)
# Create a meshgrid from x and y grid points
X, Y = np.meshgrid(x_grid, y_grid)

potential_field = np.zeros_like(X)
repulsive_potential = np.zeros_like(X)
gradient_y, gradient_x = np.gradient(potential_field,x_grid,y_grid)
gx_interp_func = interp2d(x_grid, y_grid, gradient_x, kind='linear')
gy_interp_func = interp2d(x_grid, y_grid, gradient_y, kind='linear')
epsilon=50.0


def pot_field(goal_position,repulsors): #potential field
    # For attractive field, potential field is given as constant times distance from the goal
    # For repulsive field, first a grid function is prepared with value K_obstacle for every point lying on or inside an obstacle.
    # For circles to avoid, all the grid point lying within the circle to be avoided are set to 100
    # For all the other points grid function is set to zero
    # Repulsive potential at a given point is calculate as summation over all grid of k_grid*grid_function*distance
    # Final potential field is sum of attractive and repulsive potential field
    global potential_field, obstacle_positions, x_grid, y_grid, gx_interp_func, gy_interp_func
    k_attractive=5.0
    k_obstacle=5.0
    k_grid=10.0
    max_potential=1000000000.0
    # Initialize potential field matrix
    potential_field = np.zeros_like(X)
    repulsive_potential = np.zeros_like(X)
    
    # Initialize grid with zeros
    grid2 = np.zeros_like(X)
    
    # Set values for points inside the obstacles
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            inside_obstacle = False
            for obstacle_info in obstacle_positions:
                x_min_obstacle, y_min_obstacle = obstacle_info[0], 800.0 - (obstacle_info[1] + obstacle_info[3])
                x_max_obstacle, y_max_obstacle = x_min_obstacle + obstacle_info[2], y_min_obstacle + obstacle_info[3]
                if x_min_obstacle <= X[j, i] <= x_max_obstacle and y_min_obstacle <= Y[j, i] <= y_max_obstacle:
                    grid2[j, i] = k_obstacle  # Set value to -1 if it's inside an obstacle
                    inside_obstacle = True
                    break
            if not inside_obstacle:
                grid2[j, i] = 0  # Set value to 0 if it's neither in the goal nor an obstacle
            
            if len(repulsors[0])>0:
                for repulsor in repulsors:
                    dis=np.sqrt(((X[j,i]-repulsor[0])**2)+((Y[j,i]-repulsor[1])**2))
                    if dis<=10:
                        grid2[j,i]=100
                        break
    grid=grid2    
    
    # Calculate potential field values
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            
            disX=X-(x_grid[i]*np.ones_like(X))
            disY=Y-(x_grid[j]*np.ones_like(Y))
            dis=np.sqrt((disX**2)+(disY**2))
            pot=k_grid*grid/(dis+1e-6)
            
            repulsive_potential[j,i]=np.sum(pot)
            distance_to_goal = np.linalg.norm([X[j, i] - goal_position[0], Y[j, i] - goal_position[1]])
            attractive_potential = k_attractive*distance_to_goal
            potential_field[j, i] = min(max_potential, attractive_potential + repulsive_potential[j,i])
     
    gradient_y, gradient_x = np.gradient(potential_field,x_grid,y_grid) #gradient of potential field
    
    gx_interp_func = interp2d(x_grid, y_grid, gradient_x, kind='linear')    #gradient interpolate function to calculate the gradient at a point
    gy_interp_func = interp2d(x_grid, y_grid, gradient_y, kind='linear')


    # Contour plot for diagnosis
    '''fig, ax = plt.subplots(figsize=(12,12))
    contour=ax.contourf(X, Y, potential_field, cmap='coolwarm')  # Adjust the colormap as needed
    plt.colorbar(contour, ax=ax)
    
    fig, ax1 = plt.subplots(figsize=(12,12))
    contour1=ax1.contour(X, Y, gradient_x, cmap='coolwarm')  # Adjust the colormap as needed
    plt.colorbar(contour1, ax=ax1)
    
    fig, ax2 = plt.subplots(figsize=(12,12))
    contour2=ax2.contour(X, Y, gradient_y, cmap='coolwarm')  # Adjust the colormap as needed
    plt.colorbar(contour2, ax=ax2)
    
    fig, ax3 = plt.subplots(figsize=(12,12))
    contour3=ax3.contour(X, Y, grid, cmap='coolwarm')  # Adjust the colormap as needed
    plt.colorbar(contour3, ax=ax3)'''


def goal_reached(x_curr,y_curr, goal_position, tolerance=5):    # checks if the current location is within the tolerance limit of goal
    # Calculate the Euclidean distance between current and goal positions
    distance_to_goal = np.sqrt((goal_position[0] - x_curr)**2 + (goal_position[1] - y_curr)**2)
    if distance_to_goal <= tolerance:
        return True
    else:
        return False
    
########################################################################################################################################################
# Path generation
max_iterations = 100000 #maximum number of allowable iteration to find the goal
xdes=np.zeros((max_iterations,1))   # x value on path
ydes=np.zeros((max_iterations,1))   # y value on path
tht=np.zeros((max_iterations,1))    # angle
start=[400,100]                     # start location
x_current=start[0]
y_current=start[1]
xdes[0]=start[0]
ydes[0]=start[1]
tht[0]=-0.5*np.pi

last_nonzero_index = np.where(xdes != 0)[0][-1]
last_nonzero_value = xdes[last_nonzero_index]
xdes[last_nonzero_index + 1:] = last_nonzero_value

last_nonzero_index = np.where(ydes != 0)[0][-1]
last_nonzero_value = ydes[last_nonzero_index]
ydes[last_nonzero_index + 1:] = last_nonzero_value

last_nonzero_index = np.where(tht != 0)[0][-1]
last_nonzero_value = tht[last_nonzero_index]
tht[last_nonzero_index + 1:] = last_nonzero_value

def path(goal_position):    #path planning algorithm, takes a step in the direction of gradient 
    global xdes, ydes, tht,x_current,y_current, repulsors
    
    max_iterations = 100000
    step_size=0.05

    xdes=np.zeros((max_iterations,1))
    ydes=np.zeros((max_iterations,1))
    tht=np.zeros((max_iterations,1))
    xdes[0]=start[0]
    ydes[0]=start[1]
    tht[0]=-np.pi/2;
    
    ep=0.05

    pot_field(goal_position,repulsors)
    
    stuck_flag=0
    stuck_iter=0
    stuck_itermax=2000
    
    for i in range(max_iterations):
        print(i)
        gx = gx_interp_func(x_current, y_current)
        gy = gy_interp_func(x_current, y_current)
        
        if abs(gx>10):
            gx=0.0
            
        if abs(gy>10):
            gy=0.0
        #print('i:',i)
        x_current-=step_size*gx
        y_current-=step_size*gy
        
        x_current = min(max(x_current, 7), 793)
        y_current = min(max(y_current, 7), 793)
        
        xdes[i]=x_current
        ydes[i]=y_current
        if i>1:
            tht[i]=np.arctan2(ydes[i-1]-ydes[i],-xdes[i-1]+xdes[i])

        if i>100:
            if abs(xdes[i]-xdes[i-10])<ep and abs(ydes[i]-ydes[i-10])<ep: # checks if the agent is stuck at a local minima, criteria: not moving for 10 iterations and goal is not reached
                stuck_flag=1
                stuck_iter=0
                c_new = random.randint(0, 3)
                print(x_current,y_current)
                print('Corner:',corner[c_new][0],corner[c_new][1])
                new_goal=[corner[c_new][0],corner[c_new][1]]    # Randomly selects a corner and makes it the new goal for specied number of iterations
                pot_field(new_goal,repulsors)
                
        if stuck_flag==1 and stuck_iter<stuck_itermax:
            stuck_iter+=1
        elif stuck_flag==1 and stuck_iter>=stuck_itermax:
            stuck_flag=0
            pot_field(goal_position,repulsors)
        
        #print('gx:',gx,'gy:',gy,'x_current:',x_current,'y_current:',y_current)
        if goal_reached(x_current, y_current, goal_position):
            print("Goal Reached")
            break
    #plt.scatter(xdes,ydes,s=1,color='black')
    #plt.show()

    last_nonzero_index = np.where(xdes != 0)[0][-1]
    last_nonzero_value = xdes[last_nonzero_index]
    xdes[last_nonzero_index + 1:] = last_nonzero_value
    
    last_nonzero_index = np.where(ydes != 0)[0][-1]
    last_nonzero_value = ydes[last_nonzero_index]
    ydes[last_nonzero_index + 1:] = last_nonzero_value
    
    last_nonzero_index = np.where(tht != 0)[0][-1]
    last_nonzero_value = tht[last_nonzero_index]
    tht[last_nonzero_index + 1:] = last_nonzero_value

computation_done=True
# Text Input class for command input box
class TextInputBox:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = pygame.Color('white')
        self.text = ''
        self.active = True
        self.command_history = []  # Initialize an empty list to store command history
        self.history_index = -1    # Index to track the current position in command history

    def handle_event(self, event):
        global repulsors, goal_position, computation_done, iteration, x_current, y_current
        repulsors = [()]  # Initialize repulsors list with an empty tuple by default
    
        self.active=True
    
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    if self.text.lower() == 'exit':
                        pygame.quit()
                        sys.exit()
                    else:
                        computation_done=False
                        user_input=self.text
                        current_location=[x_current, y_current]
                        create_message(thread.id, current_location, user_input) # creates message to be passed to chatgpt
                        # run gpt-3.5
                        run = client.beta.threads.runs.create(
                            thread_id=thread.id,
                            assistant_id=my_assistant.id,
                            )
                        run.status='in progress'
                        while run.status!='completed':  # wait for status to turn completed
                            time.sleep(2)
                            run = client.beta.threads.runs.retrieve(
                                thread_id=thread.id,
                                run_id=run.id
                                )
                            print(run.status)
                        messages = client.beta.threads.messages.list(
                            thread_id=thread.id
                            )
                        response = messages.data[0].content[0].text.value #extracts response
                        print(response)
                        com=response.split('$')[1]  #extracts command 
                        print(com)

                        if com.startswith('goal:'): # converts command into goal location and repulsor location
                            
                            goalc=com.split(';')[0]
                            #print('goalc:',goalc)
                            goal_coords = goalc.split(':')[1]
                            goal_coords = goal_coords.replace('(', '').replace(')', '')
                            goal_coords = goal_coords.split(',')
                            #print('goal:',goal_coords)
                            if len(goal_coords) == 2:
                                try:
                                    x_goal = float(goal_coords[0])
                                    y_goal = float(goal_coords[1])
                                    goal_position = [x_goal, y_goal]
        
                                    if 'avoid:[' in com:
                                        avoid_coords = com.split('avoid:[')[1]
                                        #print('avoid_coords:',avoid_coords)
                                        avoid_coords=avoid_coords.split(']')[0]
                                        avoid_coords = avoid_coords.split(';')
                                        repulsors = []
        
                                        for coord in avoid_coords:
                                            coord = coord.replace('(', '').replace(')', '')
                                            coord = coord.split(',')
                                            if len(coord) == 2:
                                                try:
                                                    x_repulsor = float(coord[0])
                                                    y_repulsor = float(coord[1])
                                                    repulsors.append((x_repulsor, y_repulsor))
                                                except ValueError:
                                                    print("Invalid repulsor coordinates format. Please use numbers.")
                                            else:
                                                print("Invalid number of repulsor coordinates provided.")
        
                                    # Run pot_field and path functions
                                    #print('goal:',goal_position)
                                    #print('repulsors:',repulsors)
                                    pot_field(goal_position, repulsors)
                                    path(goal_position)
                                    computation_done=True
                                    iteration=0
                                    
                                except ValueError:
                                    print("Invalid coordinates format. Please use numbers.")
                            else:
                                print("Invalid number of coordinates provided.")

                    self.command_history.append(self.text)
                    self.history_index = len(self.command_history)  # Update history index     
                    self.text = ''  # Reset text input after processing
    
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                elif event.key == pygame.K_UP:
                    # Check if there is any command history
                    if self.command_history:
                        # Update the text to the previous command
                        self.history_index = max(0, self.history_index - 1)
                        self.text = self.command_history[self.history_index]
                elif event.key == pygame.K_DOWN:
                    if self.command_history:
                        if self.history_index == len(self.command_history):
                            # If at the last command, clear the text box
                            self.text = ''
                        else:
                            self.history_index = min(len(self.command_history), self.history_index + 1)
                            self.text = self.command_history[self.history_index]
                elif event.key == pygame.K_ESCAPE:
                    # Clear text input and reset history
                    self.text = ''
                else:
                    self.text += event.unicode

    def draw(self, screen):
        # Draw the background rectangle (opaque white)
        pygame.draw.rect(screen, (255, 255, 255), self.rect)
        
        # Draw the black border
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)  # Modify the thickness if needed

        # Draw the text onto the box
        font = pygame.font.Font(None, 32)  # Choose your desired font and size
        text_surface = font.render(self.text, True, (0, 0, 0))  # Black text color
        screen.blit(text_surface, (self.rect.x + 5, self.rect.y + 5))  # Adjust the position as needed
        


# Create text input box
command_input = TextInputBox(50, 750, 700, 30)

########################################################################################################################################################
# Main loop
# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Robot Motion')
# Define font
font = pygame.font.Font(None, 18)

grid_size = 100  # Adjust this value to change the grid size
grid_color = (150, 150, 150)  # Color of the grid lines
def draw_grid():
    for x in range(0, WIDTH, grid_size):
        pygame.draw.line(screen, grid_color, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, grid_size):
        pygame.draw.line(screen, grid_color, (0, y), (WIDTH, y))

running = True
iteration=0

# main pygame loop, displaying the calculated path.
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        command_input.handle_event(event)

    # Draw everything
    screen.fill(WHITE)
    draw_grid()

    # Draw circles
    for pos, color in zip(circle_positions, circle_colors):
        pygame.draw.circle(screen, color, pos, circle_radius)

    # Draw obstacles
    for obstacle in obstacle_positions:
        pygame.draw.rect(screen, BLACK, pygame.Rect(obstacle))

    # Draw text input box
    command_input.draw(screen)
    
    if computation_done:
        robot_x = max(int(xdes[iteration]),0)
        robot_y = max(int(800-ydes[iteration]),0)
        
        #tht=np.arctan2(-ydes[i+1]+ydes[i],xdes[i+1]-xdes[i])
        robot_angle = tht[iteration]  # Angle in degrees for initial orientation (you can change this as needed)
        #draw_robot(x_current, 800-y_current, robot_angle)
        draw_robot(robot_x, robot_y, robot_angle)
        iteration=min(iteration+1,len(xdes)-2)

    # Draw Matplotlib plot
    #screen.blit(surf, (50, 100))

    # Update the display
    pygame.display.flip()
    
pygame.quit()
sys.exit()
