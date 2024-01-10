from langchain import OpenAI, LLMChain, PromptTemplate
import platform

class GPTshell:
  def __init__(self):
    self.system = platform.system()
  def text_to_command(self, prompt: str, num_outputs: int = 1):
    template_text_to_command = PromptTemplate(input_variables=["num_outputs","human_input"], template="""Convert the input to {num_outputs}"""+ self.system + """possible  command: {human_input}""")
    llm = LLMChain(llm=OpenAI(temperature=0), prompt=template_text_to_command, verbose=False)
    return llm.predict(human_input=prompt, num_outputs=num_outputs).strip()
  
  def command_to_text(self, prompt: str):
    template_command_to_text = PromptTemplate(input_variables=["human_input"], template="""Describe the """ + self.system + """ command in English Language: {human_input}""")
    llm = LLMChain(llm=OpenAI(temperature=0), prompt=template_command_to_text, verbose=False)
    return llm.predict(human_input=prompt).strip()