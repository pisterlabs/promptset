from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from .history_storage import load_conversation_history, get_chat_history, log_qna
from infrastructure.model.executor import RetrieveKakaoDataIntentExecutor, WebsearchIntentExecutor, FailbackIntentExecutor
from utils.localfile_loader import read_file

# load .env
load_dotenv()

class AutonomousAgent():
  def __init__(self, max_loop_count: int = 3):
    llm = ChatOpenAI(
      temperature=0,
      max_tokens=3000,
      model="gpt-3.5-turbo-16k")
    self.max_loop_count: int = max_loop_count
    self.guess_satisfied_qna_chain = LLMChain(
      llm=llm,
      prompt=ChatPromptTemplate.from_template(
        template=read_file("./infrastructure/model/templates/guess_satisfied_qna_template.txt"),
      ),
      output_key="intent",
      verbose=True
    )
    self.guess_intent_chain = LLMChain(
      llm=llm,
      prompt=ChatPromptTemplate.from_template(
        template=read_file("./infrastructure/model/templates/guess_intent_template.txt"),
      ),
      output_key="intent",
      verbose=True
    )
    self.executors = [
      RetrieveKakaoDataIntentExecutor(llm),
      WebsearchIntentExecutor(llm),
      FailbackIntentExecutor(llm),
    ]

  def run(self, user_message, conversation_id: str = "dummy"):
    history_file = load_conversation_history(conversation_id)
    context = self.initialize_context(user_message, conversation_id)
    
    intent_loop_count: int = 0
    wrong_answers = []
    while intent_loop_count < self.max_loop_count:
      prev_answer = context["current_answer"]
      if self.guess_qna_done(context):
        answer = prev_answer
        break
      
      intent = self.guess_intent(context)
      for executor in self.executors:
        if(executor.support(intent)):
          answer = executor.execute(context)
          context["current_answer"] = answer
          break
      intent_loop_count += 1
      wrong_answers.append(f"intent: {intent} / answer: {answer}")
      context["wrong_answers"] = "\n".join(wrong_answers)
      print(f"[SYSTEM]: loop ({intent_loop_count} / {self.max_loop_count})")
    log_qna(history_file, user_message, answer)
    return answer
  
  def initialize_context(self, user_message, conversation_id): 
    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["current_answer"] = ""
    context["wrong_answers"] = ""
    context["chat_history"] = get_chat_history(conversation_id)
    return context
  
  def guess_qna_done(self, context):
    if(context["current_answer"] == ""):
      return False
    print(f"User: " + context["user_message"])
    print(f"Assistant: " + context["current_answer"])
    response = self.guess_satisfied_qna_chain.run(context)
    is_done = response == "Y"
    print(f"[SYSTEM] response: " + response)
    print(f"[SYSTEM] Is find answer? " + str(is_done))
    return is_done
  
  def guess_intent(self, context):
    intent = self.guess_intent_chain.run(context)
    print(f"[SYSTEM] I guess, I need to do {intent}!")
    return intent