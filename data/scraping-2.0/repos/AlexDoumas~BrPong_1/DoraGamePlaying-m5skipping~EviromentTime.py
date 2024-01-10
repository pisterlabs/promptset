"""
include the gym library to the library path
this needs to happen because the installation on the mpi is not in the right place.
"""
#import sys
#sys.path.append('/home/dylopd/Documents/OpenAIGym/env/lib/python2.7/site-packages/')

"""
import the openai gym games
"""
import gym
import numpy as np
from scipy.misc import imresize
import pdb

import ObjectDetector as imageParser

"""
this environment can be either Breakout or Pong from openAI
you can choose to either get the screen image back or a vector representation of the object present on screen

you can save a gif of the last played game in this environment
"""
class Enviroment:
  
  def __init__(self,Breakout=True,gameMode="full",rewardMode="raw",autoFire=False):
    #if true play breakout if false play pong
    self.Breakout = Breakout
    #if true return screen images if false return vector representations
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
  change the game mode of the enviroment
  """
  def setGameMode(self,gameMode):
    self.gameMode = gameMode
  
  """
  perform an action in the game
  the action that is performed is the action given modulo the number of valid action_space
  this ensures that all integers are valid actions
  
  returns a gamestate representation (screen image or vector) a reward for the current action, whether not the game is finished or not
  and additional info that the game may provide
  """
  def step(self, action, oldS, oldaction):
    
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
    
    #take the current action
    S,r,done,info = self.env.step(validAction)
    #add the openai reward to the rawReward
    self.rawReward += r
    #save the screen image
    self.frames.append(S)
    
    double = False
    if self.gameMode == 'vector2':
        double = True
    ballInGame = self.ballInGame(double)
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
    #pdb.set_trace()
    #enhance the reward to make it less parse
    if(self.rewardMode == "raw"):
      return S,r,done,info
    elif(self.rewardMode == "label"):
      r = self.getLabel(S)
    elif(self.rewardMode == "bounce"):
      r = self.bounceReward(S)
    elif(self.rewardMode == "follow"):
      r = self.followReward(action,S, oldS, oldaction)
    #Use the follow reward as the evaluation function
    if(self.rewardMode == "follow"):
      self.reward += r
    else:
      self.reward += self.followReward(action,S, oldS)
    
    return S,r,done,info
  
  def preprocessGameState(self):
    S = None
    if(self.gameMode == "raw"):
      S = self.frames[-1]
    elif(self.gameMode == "vector"):
      S = self.getGamestateVector()
    elif(self.gameMode == "neural"):
      #make sure that the screen image is 84X84X4 and apply the preprocessing of the atari paper
      S = self.getNeuralInput()    
    elif(self.gameMode == "vector2"):
      S = self.getGamestateVector(double=True)
    
    return S
    
    
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
  
  #reinforcement reward structure
  def followReward(self,action,S, oldS):
    #reward = 0
    
    #if(self.Breakout):
    #  self.memory["labeledAction"] = self.getActionBreakout(S)
    #else:
    #  self.memory["labeledAction"] = self.getActionPong(S)
    #  if(self.memory["labeledAction"] == 1):
    #      self.memory["labeledAction"] = 2
    #  elif(self.memory["labeledAction"] == 2):
    #      self.memory["labeledAction"] = 1
    #if(not "labeledAction" in self.memory.keys()):
    #  reward = 0
    #elif(action == self.memory["labeledAction"]):
    #  reward =  1.0
    #else:
    #  reward = -1.0
      
    #if(self.memory["labeledAction"] != 0):
    #  self.memory["labeledAction"] += 1
    #return reward
    reward = 0
    
    if(not "labeledAction" in self.memory.keys()):
      reward = 0
    
    elif(action == self.memory["labeledAction"]):
      reward =  1.0
    
    else:
      reward = -1.0   
    
    
    if(self.Breakout):
      self.memory["labeledAction"] = self.getActionBreakout(S, oldS)
    else:
      self.memory["labeledAction"] = self.getActionPong(S)
      if(self.memory["labeledAction"] == 1):
	self.memory["labeledAction"] = 2
      elif(self.memory["labeledAction"] == 2):
	self.memory["labeledAction"] = 1
      
    if(self.memory["labeledAction"] != 0):
      self.memory["labeledAction"] += 1
    return reward
    
  
  #calculate where the ball would meet the paddle
  def getCollisionPointBreakout(self,Ball):
    if(Ball["XVel"] > 0):
      #in how many steps does the ball reach the bottom
      collitionTime = (0.79047619 - Ball["X"])/Ball["XVel"]
      #when it reaches the bottom what is its Y cordinate
      Ycollision = Ball["YVel"]*collitionTime + Ball["Y"]
      
      #if the Y cordinate is negative it means that the ball bounced on the left side
      #asuming the outcomming corner is the same as the incoming corner 
      if(Ycollision <= 0):
	Ycollision = -1*Ycollision
      elif(Ycollision >= 1.0):
	#if the ycordinate is larger than 1.0 it bounced on the right side
	Ycollision = 1.0 - (Ycollision - 1.0)
      
      return Ycollision
      
    return 0
  
  def getActionBreakout(self, gameState, lastGameState):
    action = 0
    if len(gameState) == 12:
        # use edge (i.e., X1, X2, Y1, Y2) coordinates. 
        paddle = {"X1":gameState[0],"Y1":gameState[1],"X2":gameState[2],"Y2":gameState[3], "XVel":gameState[4],"YVel":gameState[5]}
        ball = {"X1":gameState[6],"Y1":gameState[7],"X2":gameState[8],"Y2":gameState[9], "XVel":gameState[10],"YVel":gameState[11]}
        trajpaddle = {'Y11': lastGameState[1], 'Y12': lastGameState[1]}
        trajball = {'Y11': lastGameState[7], 'Y12': lastGameState[7]}
        # compute the trajectory of ball and paddle. 
        # get the y point of the ball, and compare to Y point of the paddle. Reward the paddle following (moving toward) the ball. 
        Ycollision1 = ball["Y1"]
        Ycollision2 = ball["Y2"]
        useTime = False
        if useTime:
            if start_game or 
            
            if(Ycollision1 == 0 and Ycollision2 == 0):
              action = 0
            elif(paddle["Y1"] - 0.05 > Ycollision1 and paddle["Y2"] + 0.05 > Ycollision2):
              action = 2
            elif(paddle["Y1"] - 0.05 < Ycollision1 and paddle["Y2"] + 0.05 < Ycollision2):
              action = 1
            elif (paddle["Y1"] - 0.05 < Ycollision1 and paddle["Y2"] + 0.05 > Ycollision2):
              action = 0
        else:
            if(Ycollision1 == 0 and Ycollision2 == 0):
              action = 0
            elif(paddle["Y1"] - 0.05 > Ycollision1 and paddle["Y2"] + 0.05 > Ycollision2):
              action = 2
            elif(paddle["Y1"] - 0.05 < Ycollision1 and paddle["Y2"] + 0.05 < Ycollision2):
              action = 1
            elif (paddle["Y1"] - 0.05 < Ycollision1 and paddle["Y2"] + 0.05 > Ycollision2):
              action = 0
    else:
        # use midpoint (single x and y) coordinates. 
        paddle = {"X":gameState[0],"Y":gameState[1],"XVel":gameState[2],"YVel":gameState[3]}
        ball = {"X":gameState[4],"Y":gameState[5],"XVel":gameState[6],"YVel":gameState[7]}
        # get the y point of the ball, and compare to Y point of the paddle. Reward the paddle following (moving toward) the ball. 
        #Ball = self.getBallAfterGettingPoint(ball)
        Ycollision = ball["Y"]#self.getCollisionPointBreakout(ball)
        #print Ycollision
        if(Ycollision == 0):
          action = 0
        elif(paddle["Y"] - 0.05 > Ycollision):
          action = 2
        elif(paddle["Y"] + 0.05 < Ycollision):
          action = 1

    return action
  
  #calculate where the ball would meet the paddle
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
  
  def getActionPong(self, gameState):
    action = 0
    
    paddle = {"X":gameState[0],"Y":gameState[1],"XVel":gameState[2],"YVel":gameState[3]}
    ball = {"X":gameState[4],"Y":gameState[5],"XVel":gameState[6],"YVel":gameState[7]}
    
    #Ball = self.getBallAfterGettingPoint(ball)
    Xcollision = ball["X"]#self.getCollisionPointPong(ball)
    
    if(Xcollision == 0):
      action = 0
    elif(paddle["X"] - 0.034 > Xcollision ):
      action = 2
    elif(paddle["X"] + 0.035 < Xcollision ):
      action = 1

    return action
  
  
  def getLabel(self,S):
    if(self.fullImage):
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
  reset the game enviroment
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
  
  def ballInGame(self, double):
    vector = self.getGamestateVector(double)
    if not double:
        return not (vector[4] == 0 and vector[5] == 0)
    else:
        return not (vector[6] == 0 and vector[7] == 0)
  
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
    