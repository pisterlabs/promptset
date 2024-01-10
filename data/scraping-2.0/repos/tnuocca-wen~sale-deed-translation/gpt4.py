from langchain.chat_models import ChatOpenAI
# from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
# from langchain import PromptTemplate
# from langchain.llms import OpenAI
from langchain.chains import LLMChain
def simp_clean(text):
  llm = ChatOpenAI(model = 'gpt-4', temperature=0)
  template = '''
  Analyse the given sentence relating to a land sale and see what words are wrong and out of place(these may sometimes include names and jibberish characters) in it and remove them, also correct grammar mistakes and make the sentence logically correct
  In effect clean the sentence preserving all the pertinent and important details. By no means add fake details.:" {text}"
  Response:
  '''
  prompt_template = PromptTemplate(
      input_variables=["text"],
      template=template,
  )

  chain = LLMChain(llm=llm, prompt=prompt_template)
  response = chain.run({"text":text})
  # print(response)
  return response