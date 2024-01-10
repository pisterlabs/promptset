import os
import logging
import openai

from . import chat
from . import design_chat
from . import chat_functions
from . import design_functions
from . import db_access

def get_user_input():
  return input("> ").strip()

def initializeApp():
  openai.api_key = os.environ['OPENAI_API_KEY']  
  BASE_DIR = os.getcwd()
  log_file_name = os.path.join(BASE_DIR, 'log-chat.log')
  FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
  logging.basicConfig(filename=log_file_name,
                      level=logging.INFO,
                      format=FORMAT,)

  dir = os.path.split(os.path.split(__file__)[0])[0]
  dir = os.path.join(dir, 'instance')
  design_functions.IMAGE_DIRECTORY = dir
  print(f"dir: {dir}")
  DATABASE=os.path.join(dir, 'worldai.sqlite')
  db_access.init_config(DATABASE)
  
def chat_loop():
  initializeApp()
  db = db_access.open_db()
  chat_session = design_chat.DesignChatSession()
  logging.info("\nstartup*****************");
  print("Welcome to the world builder!\n")
  print("You can create and design worlds with main characters, key sites, and special items.")
  print("")

  while True:
    try:
      user = get_user_input()
    except EOFError:
      break
    if user == 'exit':
      break
    if len(user) == 0:
      continue

    message = chat_session.chat_message(db, user)
    assistant_message = message["assistant"]
    text = message.get("updates")
    print(assistant_message)
    if len(text) > 0:
      print(text)

  #pretty_print_conversation(BuildMessages(message_history))
  print("Tokens")
  
  print("\nRunning total")
  chat_functions.dump_token_usage(db)
  db.close()

if __name__ ==  '__main__':
  chat_loop()


