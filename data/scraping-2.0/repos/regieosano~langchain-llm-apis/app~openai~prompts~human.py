from langchain.prompts import HumanMessagePromptTemplate

class Human():
	def get_human_prompt(self, human_template):
		human_prompt = HumanMessagePromptTemplate.from_template(human_template)

		return human_prompt

human = Human()