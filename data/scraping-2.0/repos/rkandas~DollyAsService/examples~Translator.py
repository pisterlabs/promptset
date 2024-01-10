from DollyClientProxy import DollyClientProxy
from langchain import PromptTemplate

ai = DollyClientProxy()
template = "### Instruction:\nTranslate \"{text}\" into {language}\n\n### Response:\n"
prompt_template = PromptTemplate(template=template, input_variables=["text", "language"])
print(ai.prompt_generate(
    prompt_template.format(text="There are very good restaurants around here.", language="French")))
print(ai.prompt_generate(
    prompt_template.format(text="Building machines with precision is most important", language="German")))
print(ai.prompt_generate(
    prompt_template.format(text="I am a good student", language="Spanish")))
