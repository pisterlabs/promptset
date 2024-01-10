from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def get_pet_name(animal_type: str,
                 pet_color: str):
    llm = OpenAI(temperature=0.7)
    prompt_template_name = PromptTemplate(
        input_variables=["animal_type", "pet_color"],
        template="I have a {animal_type} and I want a cool name for it. It is in {pet_color} color. Suggest me five cool names for my pet.",
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)
    response = name_chain({'animal_type': animal_type, 'pet_color': pet_color})
    return response


if __name__ == "__main__":
    print(get_pet_name("dog", "red"))