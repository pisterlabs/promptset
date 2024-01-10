from Module.LLMChain.CustomLLM import CustomLLM_GPT, CustomLLM_Llama, CustomLLM_FreeGPT
from Module.Proxy.gpt3 import Completion
from Module.Template.BaseTemplateForImage import text_summary_for_image_generation
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class CharacterImageGeneration:
  def image_gen(input_char_prompt):
    llm = CustomLLM_FreeGPT()
    image_generate_prompt_result = text_summary_for_image_generation()
    prompt = PromptTemplate(template=image_generate_prompt_result, input_variables=["description"])
    chain = LLMChain(llm=llm, prompt=prompt)
    question = input_char_prompt
    result = chain.run(question)
    return result