"""
include the gym library to the library path
this needs to happen because the installation on the mpi is not in the right place.
"""
import sys
#sys.path.append('/home/dylopd/Documents/OpenAIGym/env/lib/python2.7/site-packages/')

"""
Import the openai gym games. This script handles the game interface as a wrapper class for OpenAI gym. 
Here you can specify which games are played, the preprocessing of the environment and which reward is used. ATARI break out and pong
are edited so that all games have 3 actions: left, right and do nothing. 
"""
import gym
import numpy as np
from scipy.misc import imresize

import ObjectDetector as imageParser

"""
this environment can be either Breakout or Pong from openAI
you can choose to either get the screen image back or a vector representation of the object present on screen

you can save a gif of the last played game in this environment
"""
class Enviroment:
  
  def __init__(self,Breakout=True,gameMode="full",rewardMode="raw",autoFire=False):
    #if true play breakout, if false play pong
    self.Breakout = Breakout
    #if true, return screen images if false return vector representations
    self.gameMode = gameMode
    self.rewardMode = rewardMode
    self.memory = {}
    
    self.lastVector = None
    self.tempReward = 0
    self.autoFire = autoFire
    
    self.vectorState = None
    
    if(Breakout):
      self.env = gym.make("Breakout-v0")
    else:
      self.env = gym.make("Pong-v0")
    
    #a variable to save the total reward earned in the current game
    self.reward = 0.0
    
    #a variable to save the reward recieved from the openai gym
    self.rawReward = 0.0
    
    #variable to keep track of the number of actions taken in this game
    self.steps = 0
    
    #list to save frames in to make a video later
    self.frames = []
  
  """
  change the representation of the game as it is presented to the agent
  """
  def setGameMode(self,gameMode):
    self.gameMode = gameMode
  
  """
  perform an action in the game
  the action that is performed is the action given modulo the number of valid action_space
  this ensures that all integers are valid actions
  
  returns a gamestate representation (screen image or vector) a reward for the current action, whether not the game is finished or not
  and additional info that the game may provide
  
  The standard openAI gym enviroment of breakout has 4 valid actions:
  0: do nothing
  1: launch a new ball if none is in the game present at the moment, if there is already a ball present do nothing
  2: move to the right side of the screen
  3: move to the left side of the screen
  
  The openAI gym for pong has 6 valid actions:
  0: do nothing
  1: do nothing
  2: move towards the top of the screen
  3: move towards the bottom of the screen
  4: move towards the top of the screen
  5: move towards the bottom of the screen
  
  To match up the controlls for the two games this enviroment wrappers defines 3 valid actions for both games
  
  valid ations are:
  0: do nothing, the agent will not move
  1: move right, the agent will move to the right seen from the paddle looking at the ball
  2: move left, the agent will move to the left seen from the paddle looking at the ball
  
  In breakout the enviroment will automaticly launch a new ball when the ball leaves the game.
  
  This function should receive one of the 3 valid actions, and will map the given action to the correct actions as 
  defined by the openai gym
  
  All actions that are not 0,1,2 will be mapped to a valid action by taking the rest after deviding by the number of actions
  defined by the openAI gym
  """
  def step(self, action):
    
    """
    Calculate the openAI gym action from the given action
    """
    #map the agents actions to the game actions
    if(action != 0):
      action += 1
    
    if(not self.Breakout):
      #swap the action this makes sure that in both breakout and pong we are looking at the situation from the paddle to the ball and action 1 is left and action 2 is right respectively
      if(action == 3):
	action = 2
      elif(action == 2):
	action = 3
    
    #make sure that the action is valid in the current game
    validAction = action%self.env.action_space.n
    
    """
    excecute the action and save the results
    """
    
    #take the current action
    S,r,done,info = self.env.step(validAction)
    #add the openai reward to the rawReward
    self.rawReward += r
    
    #save the screen image
    self.frames.append(S)
    
    """
    if the ball is outside the game and we are playing breakout and the auto fire function is enabled 
    then launch a new ball
    """
    
    ballInGame = self.ballInGame()
    if(not ballInGame and self.autoFire and self.Breakout):
      S,r,done,info = self.env.step(1)
      #add the openai reward to the rawReward
      self.rawReward += r
    
    #there is a new state so the old vector state is invalid
    self.vectorState = None
    
    #make the vector representation if needed
    S = self.preprocessGameState()
    
    #increase the step counter
    self.steps += 1
    
    """
    compute the reward with the given reward function
    """
    if(self.rewardMode == "raw"):
      r = r
    elif(self.rewardMode == "label"):
      r = self.getLabel(S)
    elif(self.rewardMode == "bounce"):
      r = self.bounceReward(S)
    elif(self.rewardMode == "follow"):
      r = self.followReward(action,S)
    
    """
    Add the follow reward to the game evaluation reward
    The agent does not see this reward but the user gets to see it to evaluate the agents performance
    """
    #Use the follow reward as the evaluation function
    if(self.rewardMode == "follow"):
      self.reward += r
    else:
      self.reward += self.followReward(action,S)
    
    return S,r,done,info
  
  """
  Either take the gamestate as 'raw' images or vector representations. Returns the preprocessed gamestate.
  """
  def preprocessGameState(self):
    S = None
    if(self.gameMode == "raw"):
      S = self.frames[-1]
    elif(self.gameMode == "vector"):
      # get the cordinates of the ball and paddle and there velocities and put them in a vector
      # use the controids of the objects as cordinates
      S = self.getGamestateVector()
    elif(self.gameMode == "neural"):
      #make sure that the screen image is 84X84X4 and apply the preprocessing of the atari paper
      S = self.getNeuralInput()    
    elif(self.gameMode == "vector2"):
      # get the cordinates of the ball and paddle and there velocities and put them in a vector
      # use the object edges as cordinate (Xmin,Ymin,Xmax,Ymax)
      S = self.getGamestateVector(double=True)
    
    return S
    
  """
  Defines a 'bounce' reward. 
  The agent is rewarded when it bounces the ball on the paddle and punished when the ball drops out of screen.
  Returns the reward
  """
  def bounceReward(self,S):
    if(not "rewardGiven" in self.memory.keys()):
      self.memory["rewardGiven"] = False
      self.memory["punhismentGiven"] = False
      
    if(not self.gameMode=="vector"):
	V = self.getGamestateVector()
    else:
	V = S

    offset = 0
    if(not self.Breakout):
      offset = 1

    reward = 0
    if(V[6 +offset] < 0 and not self.memory["rewardGiven"] and V[4 +offset] > 0):
      reward = 1.0
      self.memory["rewardGiven"] = True
    elif(V[4 +offset] > V[0 +offset] and not self.memory["punhismentGiven"]):
      self.memory["punhismentGiven"] = True
      reward = -2.0
    elif(V[6 +offset] > 0):
      self.memory["rewardGiven"] = False
    elif(V[4 +offset] == 0):
      self.memory["punhismentGiven"] = False

    return reward
  
  """
  Defines a 'follow' reward. 
  The agent is rewarded when it takes an action that reduces the paddle's distance to the ball, otherwise it is punished.
  Which action was the correct action to take, is determined from looking at predefined labels that is computed the preevious frame. 
  If no label exists, reward is 0 (this should only happen in the first frame when there is no previous frame to compute the right action from). 
  Returns the reward
  """    
  def followReward(self,action,S):
    reward = 0
    
    if(not "labeledAction" in self.memory.keys()):
      reward = 0
    
    elif(action == self.memory["labeledAction"]):
      reward =  1.0
    
    else:
      reward = -1.0   

    if(self.Breakout):
      self.memory["labeledAction"] = self.getActionBreakout(S)
    else:
      self.memory["labeledAction"] = self.getActionPong(S)
      if(self.memory["labeledAction"] == 1):
	self.memory["labeledAction"] = 2
      elif(self.memory["labeledAction"] == 2):
	self.memory["labeledAction"] = 1
      
    if(self.memory["labeledAction"] != 0):
      self.memory["labeledAction"] += 1
    return reward
    
  
  #calculate where the ball would meet the paddle in the game breakout
  def getCollisionPointBreakout(self,Ball):
    if(Ball["XVel"] > 0):
      #in how many steps does the ball reach the bottom
      collitionTime = (0.79047619 - Ball["X"])/Ball["XVel"]
      #when it reaches the bottom what is its Y cordinate
      Ycollision = Ball["YVel"]*collitionTime + Ball["Y"]
      
      #if the Y cordinate is negative it means that the ball bounced on the left side
      #asuming the outcoming corner is the same as the incoming corner 
      if(Ycollision <= 0):
	Ycollision = -1*Ycollision
      elif(Ycollision >= 1.0):
	#if the ycoordinate is larger than 1.0 it bounced on the right side
	Ycollision = 1.0 - (Ycollision - 1.0)
      
      return Ycollision
      
    return 0
  
  '''
  This function determines which action to take in Breakout based on the current gamestate. X and Y coordinates from the ball and paddle,
  velocity of the ball and paddle are loaded from the gamestate.   
  Returns an action.
  '''
  def getActionBreakout(self, gameState):
    action = 0
    # if the cordinates are double
    if(len(gameState) == 12):
      paddle = {"X1":gameState[0],"Y1":gameState[1],"X2":gameState[2],"Y2":gameState[3],"XVel":gameState[4],"YVel":gameState[5]}
      ball = {"X1":gameState[6],"Y1":gameState[7],"X2":gameState[8],"Y2":gameState[9],"XVel":gameState[10],"YVel":gameState[11]}
      
      # when the ball is above the paddle
      if(paddle["Y1"] < ball["Y1"] and paddle["Y2"] > ball["Y2"]):
	# do nothing
	action = 0
      elif(paddle["Y1"] > ball["Y1"]):
	# move left
	action = 2
      elif(paddle["Y2"] < ball["Y2"]):
	# move right
	action = 1

    # if the cordinates are single
    elif(len(gameState) == 8):
      paddle = {"X":gameState[0],"Y":gameState[1],"XVel":gameState[2],"YVel":gameState[3]}
      ball = {"X":gameState[4],"Y":gameState[5],"XVel":gameState[6],"YVel":gameState[7]}
      
      Ycollision = ball["Y"]#self.getCollisionPointBreakout(ball)
      
      #Do nothing is ball Y coordinate is 0 meaning there is no ball
      if(Ycollision == 0):
	action = 0
      #If the (middle of the) Y coordinate of the paddle is larger than the Y coordinate of the ball, go right
      elif(paddle["Y"] - 0.05 > Ycollision):
	action = 2
      #If the (middle of the) Y coordinate of the paddle is smaller than the Y coordinate of the ball, go left
      elif(paddle["Y"] + 0.05 < Ycollision):
	action = 1
    
    return action
  
  #calculate where the ball would meet the paddle in the game of pong
  def getCollisionPointPong(self,Ball):
    if(Ball["YVel"] > 0):
      #in how many steps does the ball reach the bottom
      collitionTime = (0.8875 - Ball["Y"])/Ball["YVel"]
      #when it reaches the bottom what is its Y cordinate
      Xcollision = Ball["XVel"]*collitionTime + Ball["X"]
      
      #if the Y cordinate is negative it means that the ball bounced on the left side
      #asuming the outcomming corner is the same as the incoming corner 
      if(Xcollision <= 0):
	Xcollision = -1*Xcollision
      elif(Xcollision >= 0.9):
	#if the ycordinate is larger than 1.0 it bounced on the right side
	Xcollision = 0.9 - (Xcollision - 0.9)
      
      return Xcollision
      
    return 0
  
  '''
  This function determines which action to take in Pong based on the current gamestate. X and Y coordinates from the ball and paddle,
  velocity of the ball and paddle are loaded from the gamestate.   
  Returns an action.
  '''
  def getActionPong(self, gameState):
    action = 0
    
    # if the cordinates are double
    if(len(gameState) == 12):
      paddle = {"X1":gameState[0],"Y1":gameState[1],"X2":gameState[2],"Y2":gameState[3],"XVel":gameState[4],"YVel":gameState[5]}
      ball = {"X1":gameState[6],"Y1":gameState[7],"X2":gameState[8],"Y2":gameState[9],"XVel":gameState[10],"YVel":gameState[11]}
      
      # when the ball is above the paddle
      if(paddle["X1"] < ball["X1"] and paddle["X2"] > ball["X2"]):
	# do nothing
	action = 0
      elif(paddle["X1"] > ball["X1"]):
	# move left
	action = 2
      elif(paddle["X2"] < ball["X2"]):
	# move right
	action = 1
    
    # if the cordinates are single
    elif(len(gameState) == 8):
      paddle = {"X":gameState[0],"Y":gameState[1],"XVel":gameState[2],"YVel":gameState[3]}
      ball = {"X":gameState[4],"Y":gameState[5],"XVel":gameState[6],"YVel":gameState[7]}
      
      Xcollision = ball["X"]#self.getCollisionPointPong(ball)
      print ball["X"]
      
      if(Xcollision == 0):
	action = 0
      elif(paddle["X"] - 0.034 > Xcollision ):
	action = 2
      elif(paddle["X"] + 0.035 < Xcollision ):
	action = 1

    return action
  
  
  """
  get a vector with lenght 3 representing the optimal action for the agent to take in this state
  """
  def getLabel(self,S):
    if(not self.gameMode == "vector"):
	V = self.getGamestateVector()
    else:
	V = S
    
    action = 0
    
    if(self.Breakout):
      action = self.getActionBreakout(V)
    else:
      action = self.getActionPong(V)

    actionVector = np.zeros((1,3))
    actionVector[0,action] = 1.0
    return actionVector
    
  """
  reset the game environment
  clear the frames of the last game
  reset the game reward to zero
  and set the number of steps taken to zero
  """
  def reset(self):
    S = self.env.reset()
    self.reward = 0.0
    self.frames = [S]
    self.steps = 0
    self.rawReward = 0.0
    
    S = self.preprocessGameState()
    
    return S
  
  """
  apply the same preprocessing as in the atari2600 paper
  
  that per pixel the max of the last to frames (apparently there are object that are only visible on even frames and objects only visible on odd frames)
  
  calculate the Y channel of the screen image and add it to the screen image as a forth channel
  resize the resulting package down to an image of 84X84X4
  """
  def getNeuralInput(self):
    image = np.max(np.array(self.frames[-2:]),axis=0)
    Y = RGB2YUV(image)[:,:,0:1]
    image = np.concatenate((image,Y),axis=2)
    image = imresize(image,(84,84))
    return image
  
  """
  A function that checks wheter of not the ball is in the game
  """
  def ballInGame(self):
    vector = self.getGamestateVector()
    
    # if the cordinates of the ball are zero there is no ball found    
    if(len(vector) == 8):
      return not (vector[4] == 0 and vector[5] == 0)
    elif(len(vector) == 12):
      return not (vector[6] == 0 and vector[5] == 0 and vector[8] == 0 and vector[9] == 0)
  
  """
  extract the ball and paddle for pong and breakout and convert the image into a vector representation
  """
  def getGamestateVector(self,double=False):
    if(not self.vectorState is None):
      return self.vectorState
    
    if(self.Breakout):
      bar,barVelocity,ball,ballVelocity = imageParser.detectObjectBreakout(self.frames[-1],double=double)
    else:
      bar,barVelocity,ball,ballVelocity = imageParser.detectObjectPong(self.frames[-1],double=double)
      
    # vector content:
    # bar_X,bar_Y,Bar_XSpeed,Bar_YSpeed,ball_X,ball_Y,ball_XSpeed,ball_YSpeed
    vector = []
    vector.extend(bar)
    vector.extend(barVelocity)
    vector.extend(ball)
    vector.extend(ballVelocity)
    vector = np.array(vector)
    self.vectorState = vector
    
    if not (len(vector) == 12 or len(vector) == 8):
      print "Error the vector is not of the correct lenght"
    
    return vector
  
  """
  visualise the current game state
  """
  def render(self):
    #show the image screen of the game state
    self.env.render()
  
  """
  make a video of the last played game
  """
  def makeVideo(self,filename):
    import makeVideo as mv
    mv.makeVideo(self.frames,filename)
    
#input is a RGB numpy array with shape (height,width,3), can be uint,int, float or double, values expected in the range 0..255
#output is a double YUV numpy array with shape (height,width,3), values in the range 0..255
def RGB2YUV( rgb ):
      
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
      
    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv
  
if(__name__ == "__main__"):
  env = Enviroment(gameMode="vector2")
  S = env.reset()
  
  done = False
  for i in range(100):
    #print i
    S,R,done,info = env.step(i)
    print len(S)
    env.render()
  env.makeVideo('test.gif')
    