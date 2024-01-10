from swarmnet import SwarmNet
from openai import OpenAI
from math import pi
from threading import Lock
from typing import Optional, List, Tuple
from .grid import Grid

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

#! Will need some way of determining which command in the plan is for which agent
#! Use some ID prefixed to the command?

dl: List[Tuple[str, int]] = [("192.168.0.120", 51000), ("192.168.0.64", 51000)] # Other device
# dl: List[Tuple[str, int]] = [("192.168.0.121", 51000), ("192.168.0.64", 51000)] # Other device
# dl: List[Tuple[str, int]] = [("192.168.0.64", 51000)] # Other device

#* Update these constants
INITIALLY_THIS_AGENTS_TURN = True # Only one agent should have true
STARTING_GRID_LOC = "D1"
STARTING_GRID_HEADING = Grid.Heading.UP
ENDING_GRID_LOC = "D7"
MAX_NUM_NEGOTIATION_MESSAGES = 15

CMD_FORWARD = "@FORWARD"
CMD_BACKWARDS = "@BACKWARDS"
CMD_ROTATE_CLOCKWISE = "@CLOCKWISE"
CMD_ROTATE_ANTICLOCKWISE = "@ANTICLOCKWISE"
CMD_SUPERVISOR = "@SUPERVISOR"

LINEAR_SPEED = 0.15 # m/s
LINEAR_DISTANCE = 0.45 # m
LINEAR_TIME = LINEAR_DISTANCE / LINEAR_SPEED

ANGULAR_SPEED = 0.3 # rad/s
ANGULAR_DISTANCE = pi/2.0 # rad
ANGULAR_TIME = ANGULAR_DISTANCE / ANGULAR_SPEED

WAITING_TIME = 1

class VelocityPublisher(Node):
  def __init__(self):
    super().__init__("velocity_publisher")
    self.publisher_ = self.create_publisher(Twist, "/cmd_vel", 10)
    self.global_conv = []
    self.client: OpenAI = None
    self.max_stages = MAX_NUM_NEGOTIATION_MESSAGES
    self.this_agents_turn = INITIALLY_THIS_AGENTS_TURN
    self.other_agent_ready = False
    self.other_agent_loc = ""
    self.turn_lock = Lock()
    self.ready_lock = Lock()
    self.grid = Grid(STARTING_GRID_LOC,STARTING_GRID_HEADING, 8, 8)
  
    self.create_plan()
    
    if(len(self.global_conv) > 1):
      cmd = self.global_conv[len(self.global_conv)-1]["content"]
      for s in cmd.split("\n"):
        if(CMD_FORWARD in s):
          self.pub_forwards()
        elif(CMD_BACKWARDS in s):
          self.pub_backwards()
        elif(CMD_ROTATE_CLOCKWISE in s):
          self.pub_clockwise()
        elif(CMD_ROTATE_ANTICLOCKWISE in s):
          self.pub_anticlockwise()
        elif(CMD_SUPERVISOR in s):
          pass
        elif(s.strip() == ""):
          pass
        else:
          self.get_logger().error(f"Unrecognised command: {s}")
        self.wait_delay()
            
    self.get_logger().info(f"Full plan parsed")
        
  def _delay(self, t_target):
    t0 = self.get_clock().now()
    while(self.get_clock().now() - t0 < rclpy.duration.Duration(seconds=t_target)):
      pass
    self.get_logger().info(f"Delayed for {t_target} seconds")
    
  def linear_delay(self):
    self._delay(LINEAR_TIME)
    
  def angular_delay(self):
    self._delay(ANGULAR_TIME)
    
  def wait_delay(self):
    self._delay(WAITING_TIME)
    
  def _publish_cmd(self, msg: Twist):
    self.publisher_.publish(msg)
    self.get_logger().info(f"Publishing to /cmd_vel")
  
  def _publish_zero(self):
    self.get_logger().info(f"Zero velocity requested")
    msg = Twist()
    
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0
    
    self._publish_cmd(msg)
    
  def _pub_linear(self, dir: int):
    msg = Twist()
    
    msg.linear.x = dir * LINEAR_SPEED #? X, Y or Z?
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0
    
    self._publish_cmd(msg)    
    self.linear_delay()
    self._publish_zero()
  
  def _pub_rotation(self, dir: float):
    msg = Twist()
    
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = dir * ANGULAR_SPEED #? X Y or Z
    
    self._publish_cmd(msg)
    self.angular_delay()
    self._publish_zero()
    
  def pub_forwards(self):
    self.get_logger().info(f"Forwards command")
    self.grid.forwards()
    self._pub_linear(1)
    
  def pub_backwards(self):
    self.get_logger().info(f"Backwards command")
    self.grid.backwards()
    self._pub_linear(-1)
    
  def pub_anticlockwise(self):
    self.get_logger().info(f"Anticlockwise command")
    self.grid.anticlockwise()
    self._pub_rotation(1)
    
  def pub_clockwise(self):
    self.get_logger().info(f"Clockwise command")
    self.grid.clockwise()
    self._pub_rotation(-1)
    
  def create_plan(self):
    self.get_logger().info(f"Initialising SwarmNet")
    self.sn_ctrl = SwarmNet({"LLM": self.llm_recv, "READY": self.ready_recv, "FINISHED": self.finished_recv, "INFO": self.info_recv}, device_list = dl) #! Publish INFO messages which can then be subscribed to by observers
    self.sn_ctrl.start()
    self.get_logger().info(f"SwarmNet initialised") 
    self.sn_ctrl.send("INFO SwarmNet initialised successfully")
    
    while(not self.is_ready()):
      self.sn_ctrl.send(f"READY {self.grid}")
      self.get_logger().info("Waiting for an agent to be ready")
      self.wait_delay()
      
    self.sn_ctrl.send(f"READY {self.grid}")
      
    self.sn_ctrl.clear_rx_queue()
    self.sn_ctrl.send("INFO Agents ready for negotiation")
        
    self.client = OpenAI() # Use the OPENAI_API_KEY environment variable
    self.global_conv = [
      {"role": "system", "content": f"You and I are wheeled robots, and can only move forwards, backwards, and rotate clockwise or anticlockwise.\
        We will negotiate with other robots to navigate a path without colliding. You should negotiate and debate the plan until all agents agree.\
          You cannot go outside of the grid. Once this has been decided you should call the '\f{CMD_SUPERVISOR}' tag at the end of your plan and print your plan in a concise numbered list using only the following command words:\
            - '{CMD_FORWARD}' to move one square forwards\
            - '{CMD_BACKWARDS}' to move one square backwards \
            - '{CMD_ROTATE_CLOCKWISE}' to rotate 90 degrees clockwise \
            - '{CMD_ROTATE_ANTICLOCKWISE}' to rotate 90 degrees clockwise \
            The final plan should be a numbered list only containing these commands."}]
    self.negotiate()
    self.sn_ctrl.send("INFO Negotiation finished")
    self.sn_ctrl.kill()
    
  def is_my_turn(self):
    self.turn_lock.acquire()
    b = self.this_agents_turn
    self.turn_lock.release()
    return b

  def toggle_turn(self):
    self.turn_lock.acquire()
    self.this_agents_turn = not self.this_agents_turn
    self.turn_lock.release()
    
  def set_turn(self, b):
    self.turn_lock.acquire()
    self.this_agents_turn = b
    self.turn_lock.release()
    
  def send_req(self):
    completion = self.client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=self.global_conv,
      max_tokens=750
    )

    # print(completion.choices[0].message)
    self.global_conv.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    self.sn_ctrl.send(f"LLM {completion.choices[0].message.role} {completion.choices[0].message.content}")
    
  def toggle_role(self, r: str):
    if r == "assistant":
      return "user"
    elif r == "user":
      return "assistant"
    else:
      return ""
    
  def plan_completed(self):
    self.get_logger().info(f"Plan completed:")
    for m in self.global_conv:
      self.get_logger().info(f"{m['role']}: {m['content']}")
      
    self.sn_ctrl.send("FINISHED")
    
    while(not (self.sn_ctrl.rx_queue.empty() and self.sn_ctrl.tx_queue.empty())):
      self.get_logger().info("Waiting for message queues to clear")
      self.wait_delay()
    
    self.generate_summary()
    
  def generate_summary(self):
    self.global_conv.append({"role": "user", "content": f"Generate a summarised numerical list of the plan for the steps that I should complete. Use only the commands:\
      - '{CMD_FORWARD}' to move one square forwards\
      - '{CMD_BACKWARDS}' to move one square backwards \
      - '{CMD_ROTATE_CLOCKWISE}' to rotate 90 degrees clockwise \
      - '{CMD_ROTATE_ANTICLOCKWISE}' to rotate 90 degrees clockwise "})
    
    completion = self.client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=self.global_conv,
      max_tokens=750
    )

    self.global_conv.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    self.sn_ctrl.send(f"INFO Final plan for {self.sn_ctrl.addr}: {completion.choices[0].message.content}")
  
  def info_recv(self, msg: Optional[str]) -> None:
    pass
  
  def finished_recv(self, msg: Optional[str]) -> None:
    self.generate_summary()
  
  def llm_recv(self, msg: Optional[str]) -> None: 
    m = msg.split(" ", 1) # Msg are LLM ROLE CONTENT
    r = m[0]
    c = m[1]
    self.global_conv.append({"role": self.toggle_role(r), "content": c})
    self.toggle_turn()

  def ready_recv(self, msg: Optional[str]) -> None:
    self.ready_lock.acquire()
    self.other_agent_ready = True
    self.other_agent_loc = msg
    self.ready_lock.release()
  
  def is_ready(self):
    self.ready_lock.acquire()
    b = self.other_agent_ready
    self.ready_lock.release()
    return b

  def negotiate(self):
    current_stage = 0
    
    if self.this_agents_turn:
      self.global_conv.append({"role": "user", "content": f"I am at {self.grid}, you are at {self.other_agent_loc}. I must end at {self.other_agent_loc} and you must end at {self.grid}"})
    else:
      current_stage = 1
    
    while(current_stage < self.max_stages):
      while(not self.is_my_turn()): # Wait to receive from the other agent
        if(len(self.global_conv) > 0 and self.global_conv[len(self.global_conv)-1]["content"].rstrip().endswith(f"{CMD_SUPERVISOR}")):
          break;
        
        self.wait_delay()
        self.get_logger().info(f"Waiting for a response from another agent")
        
      # if(len(self.global_conv) > 0 and self.global_conv[len(self.global_conv)-1]["content"].rstrip().endswith(f"{CMD_SUPERVISOR}")):
      #   self.get_logger().info(f"Content ends with {CMD_SUPERVISOR}")
      #   break;
      
      self.send_req()
      self.toggle_turn()
      current_stage += 2 # Shares the current_stage
      self.get_logger().info(f"Stage {current_stage}")
      self.sn_ctrl.send(f"INFO Negotiation stage {current_stage}")
      self.get_logger().info(f"{self.global_conv}");
        
    self.plan_completed()
    current_stage = 0
  

def main(args=None):
  
  rclpy.init()
  velocity_publisher = VelocityPublisher()
  
  #* Move this logic into the node itself
  
  # global global_conv
  
  # global_conv = [
  #   {"role": "system", "content": f"@FORWARD"}]
  rclpy.spin_once(velocity_publisher) #* spin_once will parse the given plan then return
  velocity_publisher.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
    main()
