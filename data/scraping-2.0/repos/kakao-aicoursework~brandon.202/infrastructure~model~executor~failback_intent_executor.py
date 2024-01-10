from utils.localfile_loader import read_file
from .intent_executor import IntentExecutor
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate


class FailbackIntentExecutor(IntentExecutor):
  def __init__(self, llm):
    self.chain = LLMChain(
      llm=llm,
      prompt=ChatPromptTemplate.from_template(
        template=read_file("./infrastructure/model/templates/failback_response_template.txt"),
      ),
      output_key="output",
    )

  def support(self, intent):
    return True

  def execute(self, context):
    return self.chain.run(context)
