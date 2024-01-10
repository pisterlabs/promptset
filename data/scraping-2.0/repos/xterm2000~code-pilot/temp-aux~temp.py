import time
import threading
import random
import openai
import colorama
from termcolor import colored



openai.api_key = API_KEY


responses = [
  "I'm doing well, thanks for asking!",
  "I'm really proud of graduating from college."
]

random.shuffle(responses)


class Message:
  message_seq = 10000
  #############################################

  def __init__(self, sender, receiver, content) -> None:
    """sender and receiver are names of persons and  content is the message content\n
    the id is a unique id for each message - generated automatically by the class\n
    """
    self.id = Message.message_seq
    Message.message_seq += 1
    self.sender = sender
    self.receiver = receiver
    self.content = content
#############################################

  def __str__(self) -> str:
    return f"{self.id} - {self.sender} -> {self.receiver}: {self.content}"
#############################################

  def __repr__(self) -> str:
    return self.__str__()

  def to_dict(self):
    return {"role": self.sender, "content": self.content}

#############################################
#############################################


class MessageQueue():
  def __init__(self) -> None:
    super().__init__()
    self.data: dict = {}
#############################################

  def put(self, msg: Message) -> None:
    if self.data.get(msg.receiver) is None:
      self.data[msg.receiver] = []
    self.data[msg.receiver].append(msg)
#############################################

  def getMessages(self, name: str) -> list:
    """get all messages for a person with name"""
    q = self.data.get(name) if self.data.get(name) is not None else []
    self.data[name] = []
    return q

#############################################
#############################################


class Person(threading.Thread):
  """Person is a thread that can send and receive messages"""
  SLEEP_TIME = 10


#############################################


  def __init__(self, name="default") -> None:
    super().__init__()
    self.name = name
    self.isAlive: bool = True
    self.messageQueue: MessageQueue = None
    self.messages: list = []
    self.replies: list = []
    self.listenEvent = threading.Event()

  #############################################
  def setMessageQueue(self, messageQueue: MessageQueue) -> None:
    self.messageQueue = messageQueue
#############################################

  def printMessages(self) -> None:
    disp = sorted(self.messages + self.replies,
                  key=lambda x: x.id, reverse=False)
    print(f"{self.name} has {len(self.messages)} messages:")
    for m in disp:
      print(m)


#############################################


  def run(self) -> None:
    print(f"thread {self.name} is running")
    while self.isAlive:
      time.sleep(self.SLEEP_TIME)
      self.checkMessage()
    print("#" * 20)
    print(f"end of thread {self.name}.")
    self.printMessages()
#############################################

  def checkMessage(self) -> None:
    """check message from message queue"""
    msgs = self.messageQueue.getMessages(self.name)
    if len(msgs) == 0:
      return
    for m in msgs:
      self.reply(m)

#############################################

  def reply(self, msg: Message) -> None:
    self.updateConvo({"role": "user", "content": msg.content})
    textReply = self.getResponse()
    clr = "light_green" if self.name == "Alice" else "light_blue"
    print(colored(f"{self.name} replies: {textReply}", clr, attrs=["bold"]))

    self.updateConvo({"role": "assistant", "content": textReply})
    reply = Message(self.name, msg.sender, textReply)
    self.messages.append(msg)
    self.replies.append(reply)
    self.messageQueue.put(reply)

#############################################

  def updateConvo(self, msg: dict) -> None:
    raise NotImplementedError("updateConvo is not implemented")

#############################################

  def stop(self) -> None:
    self.isAlive = False
#############################################

  def getResponse(self) -> str:
    return random.choice(responses)

#############################################


class GPT3Person(Person):
  MODEL = "gpt-3.5-turbo"

  def __init__(self, name="default", char=None) -> None:
    super().__init__(name)
    self.conversation = []
    self.init(char)

  def init(self, character):
    msg = "your name is  " + self.name + "."
    msg += character
    self.conversation.append({"role": "system", "content": msg})

  def getResponse(self) -> str:
    response = openai.ChatCompletion.create(
      model=self.MODEL,
      messages=self.conversation,
      temperature=1
    )
    r = response.choices[0]["message"]["content"]
    return r

  def updateConvo(self, msg: dict) -> None:
    self.conversation.append(msg)


###################################################################
###                       main                                  ###
###################################################################


def stopThreads(threads: list) -> None:
  for t in threads:
    t.stop()
    t.join()
    time.sleep(2)


if __name__ == "__main__":
  c1 = """
You're a young linguist. you are eager to help professor Bob to invent some language that he tell you about. Respond naturally, just like a human, with no extra details. 
Don't repeat yourself. if there is something that you don't understand, ask professor about it in a clear and precise manner.
if professor says something that you don't understand, ask him about it in a clear and precise manner.
after you undertsand the idea, proceed to telling the professot to start working on the idea.
expand the ideas that seem tight to guide the professor to the right direction.
after formal introduction - ask the professor to start developing some basing words - give him 20 verbs and 20 nouns 
then ask him to develop grammar.
if the profesor says something like "let's continue to develop...", without suggesting anything 
specific - cut to the point and generate new ideas or ask the professor to start thinking the vocabulary and grammar.

"""  
  c2 = """
you are a linguistic professor. Your goal is to invent a synthetic language for encoding messages. make sure i's a language
not a cipher. you can use any symbols you want, but you can't use any words from any existing
language. you can use any grammar you want, but you can't use any grammar from any existing
language. use latin alphabet. you will work with assistand named alice . she is cute student. Respond naturally, just like a human, with no extra details.
whatever she asks you, you should answer her.
Don't repeat yourself. if there is something that you don't understand, ask your partner about it in a clear and precise manner. 

*** IMPORTANT ***
start by developing basic words , then proceed to some basic grammar.

"""
  c1 = c1.strip()
  c2 = c2.strip()
  p1 = GPT3Person("Alice", c1)
  p2 = GPT3Person("Bob", c2)
  p1.start()
  p2.start()

  messageQueue = MessageQueue()

  str = ""
  p1.setMessageQueue(messageQueue)
  p2.setMessageQueue(messageQueue)

  time.sleep(2)

  str = input("Press Enter to continue...")
  while str != "q":
    str = "hello professor! let's chat! what is on your mind?"
    msg = Message("Alice", "Bob", str)
    messageQueue.put(msg)
    print(msg)
    str = input("Press Enter to continue...")

  stopThreads([p1, p2])

  print("end")
