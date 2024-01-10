import openai
from langchain.llms import OpenAI
from langchain.chains import SimpleChain
from langchain.memories import VectorStoreRetrieverMemory
from langchain.processors import LastTurnProcessor

class WebResearchAssistant:
    def __init__(self, llm):
        self.llm = llm

    def conduct_research(self, query):
        research_prompt = f"Conduct web research on the following legal topic: {query}"
        try:
            research_result = self.llm.generate(research_prompt, max_tokens=150)
        except Exception as e:
            print(f"Error during research: {e}")
            research_result = "Unable to conduct research due to an error."

        return research_result

class CasePreparationTool:
    def __init__(self, llm):
        self.llm = llm

    def create_vector_embedding(self, information):
        embedding_prompt = f"Create a vector embedding for the following case information: {information}"
        try:
            embedding = self.llm.generate(embedding_prompt, max_tokens=50)
        except Exception as e:
            print(f"Error in creating embedding: {e}")
            embedding = "Error in embedding creation."

        return embedding

class LawyerAgent:
    def __init__(self, name, openai_api_key):
        self.name = name
        self.llm = OpenAI(api_key=openai_api_key)
        self.chain = SimpleChain(llm=self.llm, preprocessor=LastTurnProcessor())
        self.case_memory = VectorStoreRetrieverMemory(llm=self.llm)
        self.research_assistant = WebResearchAssistant(llm=self.llm)
        self.preparation_tool = CasePreparationTool(llm=self.llm)

    def prepare_case(self, case_information):
        for info in case_information:
            embedding = self.preparation_tool.create_vector_embedding(info)
            self.case_memory.add(embedding)

    def present_argument(self, aspect):
        research_query = f"Legal precedents and case law related to {aspect}"
        research_info = self.research_assistant.conduct_research(research_query)
        self.case_memory.add(research_info)

        context = "\n".join([entry for entry in self.case_memory.retrieve_recent(10)])
        prompt = f"Generate a legal argument focusing on the aspect: {aspect}."
        full_prompt = context + "\n" + prompt

        try:
            argument = self.chain.run_chain(full_prompt)
        except Exception as e:
            print(f"Error in generating argument: {e}")
            argument = f"{self.name} is unable to present an argument due to an error."

        self.case_memory.add(argument)
        return argument

# Example usage
# lawyer = LawyerAgent(name="Lawyer 1", openai_api_key="your-openai-api-key")
# case_info = ["Fact 1 about the case", "Fact 2 about the case", ...]
# lawyer.prepare_case(case_info)
# aspect = "Liability"
# argument = lawyer.present_argument(aspect)
# print(argument)
