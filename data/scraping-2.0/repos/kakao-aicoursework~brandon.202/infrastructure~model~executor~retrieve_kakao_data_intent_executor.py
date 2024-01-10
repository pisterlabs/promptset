from utils.localfile_loader import read_file
from .vectorstore import query_on_chroma
from .intent_executor import IntentExecutor
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate


class RetrieveKakaoDataIntentExecutor(IntentExecutor):
  def __init__(self, llm):
    self.chain = LLMChain(
      llm=llm,
      prompt=ChatPromptTemplate.from_template(
        template=read_file("./infrastructure/model/templates/retrieve_template.txt"),
      ),
      output_key="output",
    )

  def support(self, intent):
    return intent == "retrieve_kakao_data"

  def execute(self, context):
    context["retrieve_result"] = query_on_chroma(context["user_message"])
    return self.chain.run(context)
