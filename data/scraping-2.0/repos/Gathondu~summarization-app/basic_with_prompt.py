from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from utils import load_env

if __name__ == "__main__":
    load_env()

    template = """
    Write a consice and short summary of the following text:
    TEXT: `{text}`
    Translate the summary to {language}
    """
    prompt = PromptTemplate(input_variables=["text", "language"], template=template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt)
    print("Write Quit or Exit to exit.")
    while True:
        text = input("What text do you want to summarize? \n")
        if text == "":
            print("Please provide text to summarize!")
            continue
        elif text.lower() in ["quit", "exit"]:
            print("Quiting application.")
            break
        language = input("In what language should the summary be in? defaults to english. \n") or "en"
        summary = chain.run({"text": text, "language": language})
        print(f'SUMMARY \n {"-"*50} \n {summary} \n')
