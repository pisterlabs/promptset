from langchain.memory import ConversationBufferMemory, FileChatMessageHistory


def load_conversation_history(conversation_id: str):
  file_path = f"./persistence/histories/{conversation_id}.json"
  return FileChatMessageHistory(file_path)

def log_user_message(history: FileChatMessageHistory, user_message: str):
  history.add_user_message(user_message)


def log_bot_message(history: FileChatMessageHistory, bot_message: str):
  history.add_ai_message(bot_message)

def log_qna(history: FileChatMessageHistory, user_message: str, bot_message: str):
  log_user_message(history, user_message)
  log_bot_message(history, bot_message)

def get_chat_history(conversation_id: str):
  history = load_conversation_history(conversation_id)
  memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="user_message",
    chat_memory=history,
  )

  return memory.buffer
