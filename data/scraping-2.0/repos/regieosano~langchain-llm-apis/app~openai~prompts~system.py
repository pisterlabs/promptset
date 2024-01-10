from langchain.prompts import SystemMessagePromptTemplate

class System():

	def get_system_prompt(self, system_template):
		system_prompt = SystemMessagePromptTemplate.from_template(system_template)

		return system_prompt


system = System()
