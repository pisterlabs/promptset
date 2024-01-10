from config import settings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

class Messages:
  def __init__(self, initial_system_message):
    self.list = [("system", initial_system_message)]
    self.messages_count = 0
    self.messages_limit = 8
    self.messages_to_remove = 6

  def get_list(self):
    return self.list

  def append(self, message):
    if message[0] == "ai":
      self._remove_context()
    self.list.append(message)
    if message[0] in ["user", "ai"]:
      self.messages_count += 1
    if self.messages_count == self.messages_limit:
      self._summarize_messages()
      self.messages_count = self.messages_limit - self.messages_to_remove

  def extend(self, messages):
    self.list.extend(messages)

  def insert_context(self, context):
    self.append(("system", context))

  def _remove_context(self):
    for message in self.list:
      if message[0] == "system" and message[1].find("[document context]") != -1:
        self.list.remove(message)

  def _get_previous_summary(self):
    for message in self.list:
      if message[0] == "system" and message[1].find("[summary]") != -1:
        return message[1].replace("[summary]", "")
    return "None"

  def _get_conversation(self, count):
    conversation = ""
    counter = count
    index = 0
    while index < len(self.list) and counter > 0:
        message = self.list[index]
        if message[0] == "user" or message[0] == "ai":
            conversation += message[0] + ": " + message[1] + "\n"
            counter -= 1
            self.list.pop(index)
        else:
            index += 1
    return conversation


  def _summary_exists(self):
    for message in self.list:
      if message[0] == "system" and message[1].find("[summary]") != -1:
        return True
    return False

  def _summary_exists(self):
    if self.list[1][0] == "system" and self.list[1][1].find("[summary]") != -1:
      return True
    return False

  def _update_summary(self, summary):
    if self._summary_exists():
      self.list[1] = ("system", "[summary]" + summary)
    else:
      self.list.insert(1, ("system", "[summary] Summary of previous conversations:\n" + summary))

  def _summarize_messages(self):
    prompt = ChatPromptTemplate.from_template("Summarize the conversation based on the given list of messages given below into one paragraph. Keep it concise, including only essential information to maintain a natural conversational flow. If there's a previous summary, incorporate the new one, following the same concise and accurate style in one paragraph.\n\nPrevious summary:\n{previous_summary}\n\nConversation:\n{conversation}")
    model = ChatOpenAI(
      openai_api_key=settings.OPENAI_API_KEY,
      model_name=settings.OPENAI_MODEL_NAME,
      temperature=settings.OPENAI_TEMPERATURE
    )
    chain = prompt | model | StrOutputParser()
    previous_summary = self._get_previous_summary()
    conversation = self._get_conversation(self.messages_to_remove)
    summary = chain.invoke(
      {"previous_summary": previous_summary, "conversation": conversation}
    )

    self._update_summary(summary)
