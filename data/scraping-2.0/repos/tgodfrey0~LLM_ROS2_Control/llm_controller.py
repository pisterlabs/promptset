from swarmnet import SwarmNet
from openai import OpenAI
from threading import Lock
from time import sleep

global_conv = []
client: OpenAI = None
max_stages = 10
this_agents_turn = True
tl = Lock()

def is_my_turn():
  tl.acquire()
  b = this_agents_turn
  tl.release()
  return b

def toggle_turn():
  global this_agents_turn
  tl.acquire()
  this_agents_turn = not this_agents_turn
  tl.release()
  
def send_req():
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=global_conv,
    # max_tokens=500
  )

  # print(completion.choices[0].message)
  global_conv.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
  sn_ctrl.send(f"LLM {completion.choices[0].message.role} {completion.choices[0].message.content}")
  
def get_api_key() -> str:
  with open("openai_key", "r") as f:
    return f.readline().rstrip()
  
def toggle_role(r: str):
  if r == "assistant":
    return "user"
  elif r == "user":
    return "assistant"
  else:
    return ""
  
def plan_completed():
  print("Plan completed:")
  # map(lambda m : print(f"{m.role}: {m.content}"), global_conv)
  for m in global_conv:
    print(f"{m['role']}: {m['content']}")
  
def llm_recv(msg: str) -> None: 
  m = msg.split(" ", 1) # Msg are LLM ROLE CONTENT
  r = m[0]
  c = m[1]
  global_conv.append({"role": toggle_role(r), "content": c}) #! Don't think this is adding to the list
  toggle_turn()
  # if("@SUPERVISOR" not in c):
  #   send_req(client)
  # else:
  #   plan_completed() #? This may have issues with only one agent finishing. Could just add a SN command

def negotiate():
  current_stage = 0
  
  if this_agents_turn:
    global_conv.append({"role": "user", "content": "I am at D1, you are at D7. I must end at D7 and you must end at D1"})
  
  while(current_stage < max_stages or not global_conv[len(global_conv)-1]["content"].endswith("@SUPERVISOR")):
    while(not is_my_turn()): # Wait to receive from the other agent
      sleep(0.5)
      print("waiting")
    
    send_req()
    toggle_turn()
    current_stage += 1
    print(f"Stage {current_stage}")
    print(global_conv);
      
  plan_completed()
  current_stage = 0

if __name__=="__main__":
  sn_ctrl = SwarmNet({"LLM": llm_recv}, device_list = [("192.168.0.120", 51000)])
  sn_ctrl.start()
  print("Communications initialised")
  input("Press any key to start")
  client = OpenAI()
  global_conv = [
    {"role": "system", "content": "You and I are wheeled robots, and can only move forwards, backwards, and rotate clockwise or anticlockwise.\
      We will negotiate with other robots to navigate a path without colliding. You should negotiate and debate the plan until all agents agree.\
        Once this has been decided you should call the '@SUPERVISOR' tag at the end of your plan and print your plan in a concise numbered list using only the following command words:\
          - 'FORWARDS' to move one square forwards\
          - 'BACKWARDS' to move one square backwards\
          - 'CLOCKWISE' to rotate 90 degrees clockwise\
          - 'ANTICLOCKWISE' to rotate 90 degrees clockwise\
          "}]
  # res = send_req(client)
  # print(res.content)
  # sn_ctrl.send(f"LLM {res.role} {res.content}")
  negotiate()
  input("Press any key to finish")
  plan_completed()
  sn_ctrl.kill()