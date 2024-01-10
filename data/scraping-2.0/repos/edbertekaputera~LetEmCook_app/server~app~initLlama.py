from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
import json

# Initializes Llama
def init_llama(model_path: str):
	B_INST, E_INST = "[INST]", "[/INST]"
	B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

	CUSTOM_SYSTEM_PROMPT="You are now a professional chef that has expertise in recipe generations"
	RULES = "Make sure you show the entire procedure for cooking it. Make sure that any extra ingredients you include are commonly found at home. Remember to include the Dish Name header."
	FORMATTING = "The structure of the recipe shall be split to five sections in the following format. Dish Name:\n Cooking Time:\n Calories:\n Protein:\n Ingredients\n Instructions\n. Make sure the ingredients appear in bullet format with asterisks. Make sure the instructions appear in numbered format. Make sure the calories and protein are given per serving"
	SYSTEM_PROMPT = B_SYS + CUSTOM_SYSTEM_PROMPT + RULES + FORMATTING + E_SYS
	instruction = "Recommend me three different healthy and high-protein recipe that are ideal for athletes and is easy to make with the following ingredients:\n'{ingredients}'"

	template = B_INST + SYSTEM_PROMPT + instruction + E_INST

	# template for an instruction with input
	prompt_with_context = PromptTemplate(
		input_variables=["ingredients"],
		template= template)
	
	llm = LlamaCpp(
		model_path=model_path,
		n_gpu_layers=1,
		n_batch=2048,
		temperature=0.72,
		n_ctx=2048,
		max_tokens=2048,
		top_p=0.74,
		top_k=0,
		f16_kv=True,
		verbose=True,
	) # type: ignore

	llm_context_chain = LLMChain(llm=llm, prompt=prompt_with_context)

	return llm_context_chain

# Initialize json from file
def init_json(file_path:str) -> dict[str, str]:
	with open(file_path, mode="r") as file:
		data = json.load(file)
	return data