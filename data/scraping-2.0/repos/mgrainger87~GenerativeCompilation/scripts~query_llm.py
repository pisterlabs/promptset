from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

def extract_assembly(output: str) -> str:
	# Find the start and end of the assembly code using the provided markers
	start_marker = "---ASSEMBLY BEGIN---"
	end_marker = "---ASSEMBLY END---"
	
	start_index = output.find(start_marker)
	end_index = output.find(end_marker)
	
	# Check if both markers are present
	if start_index == -1 or end_index == -1:
		return None
	
	# Extract the assembly code
	assembly_code = output[start_index + len(start_marker):end_index].strip()
	
	# Remove triple backticks if present
	if assembly_code.startswith("```assembly"):
		assembly_code = assembly_code[len("```assembly"):].strip()
	if assembly_code.endswith("```"):
		assembly_code = assembly_code[:-3].strip()
	
	return assembly_code

class LLMQuerier:
	def __init__(self):
		self.llm = OpenAI(
		  model_name="gpt-4",
		  temperature=0.9)
		  
		template = """The following is a conversation between a helpful and logical assistant who is skilled at generating computer assembly language and are working with a human to generate assembly language in a conversation style. When asked,the AI generates arm64 assembly that compiles on macOS. It explains its thought process as much as possible before generating the assembly. It marks the beginning and end of the final generated assembly with lines containing ---ASSEMBLY BEGIN--- and ---ASSEMBLY END--- respectively. When making changes, it always outputs the full complete assembly.
		
		Current conversation:
		{history}
		Human: {input}
		AI:"""
		PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
		  
		self.conversation = ConversationChain(
			llm=self.llm, verbose=True, prompt=PROMPT, memory=ConversationBufferMemory()
		)
	
	def performQuery(self, input):
		return extract_assembly(self.conversation.predict(input=input))
		
		

