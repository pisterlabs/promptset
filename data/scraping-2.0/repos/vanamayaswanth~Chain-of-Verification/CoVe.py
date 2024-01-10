import getpass
import spacy
from typing import List, Dict, Any
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory
from pydantic import BaseModel

# Initialize spaCy
nlp = spacy.load("en_core_web_lg")

# Custom Memory class for storing information about entities
class SpacyEntityMemory(BaseMemory, BaseModel):
    entities: dict = {}
    memory_key: str = "entities"

    def clear(self):
        self.entities = {}

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        doc = nlp(inputs[list(inputs.keys())[0]])
        entities = [
            self.entities[str(ent)] for ent in doc.ents if str(ent) in self.entities
        ]
        return {self.memory_key: "\n".join(entities)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        text = inputs[list(inputs.keys())[0]]
        doc = nlp(text)
        for ent in doc.ents:
            ent_str = str(ent)
            if ent_str in self.entities:
                self.entities[ent_str] += f"\n{text}"
            else:
                self.entities[ent_str] = text

# Securely input the OpenAI API key
OPENAI_API_KEY = getpass.getpass("OpenAI API Key:")

# Initialize OpenAI instance with the API key
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

# Step 1: Draft an initial response
initial_response_template = """Here is the {question}\n\nInitial Response:"""
initial_response_prompt = PromptTemplate(input_variables=["question"], template=initial_response_template)
draft_initial_response_chain = LLMChain(llm=llm, prompt=initial_response_prompt, output_key="output_initial_response")

# Step 2: Plan verification questions
plan_verification_questions_template = """{initial_response}\n\nPlan Verification Questions:"""
plan_verification_questions_prompt = PromptTemplate(
    input_variables=["initial_response"], template=plan_verification_questions_template
)
plan_verification_questions_chain = LLMChain(
    llm=llm, prompt=plan_verification_questions_prompt, output_key="output_verification_questions"
)

# Step 3: Answer verification questions
answer_verification_template = """Verification Question: {verification_question}\n\nAnswer: {answer}\n\n"""
answer_verification_prompt = PromptTemplate(
    input_variables=["verification_question", "answer"], template=answer_verification_template
)
answer_verification_chain = LLMChain(
    llm=llm, prompt=answer_verification_prompt, output_key="output_verification_responses"
)

# Step 4: Generate final verified response
generate_verified_response_template = """{verification_answers}\n\nFinal Verified Response:"""
generate_verified_response_prompt = PromptTemplate(
    input_variables=["verification_answers"], template=generate_verified_response_template
)
generate_verified_response_chain = LLMChain(
    llm=llm, prompt=generate_verified_response_prompt, output_key="output_final_response"
)

# Create the chain of verification
overall_chain = SimpleSequentialChain(
    chains=[
        draft_initial_response_chain,
        plan_verification_questions_chain,
        answer_verification_chain,
        generate_verified_response_chain,
    ],
    verbose=True,
    memory=SpacyEntityMemory(memory_key="entities"),  # Use the custom memory
)

# Function to run the chain of verification
def chain_of_verification(question):
    return overall_chain.run({"question": question})

if __name__ == "__main__":
    question = input("Enter a question: ")
    answer = chain_of_verification(question)
    print(answer)
