from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv


prompt = PromptTemplate(
    input_variables=["prompt"],
    template="You are a helpful assistant. {prompt}",
)

def main():
    load_dotenv()

    llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-1-pythia-12b")
    chain = LLMChain(llm=llm, prompt=prompt)
    print("OpenAssistant/oasst-sft-1-pythia-12b")

    while True:
        user_input = input("> ")
        assistant_response =chain.run(user_input)
        print("\nAssistant:\n", assistant_response, "\n")


if __name__ == '__main__':
    main()