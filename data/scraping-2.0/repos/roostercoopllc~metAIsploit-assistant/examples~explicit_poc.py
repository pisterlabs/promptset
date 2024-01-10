from tqdm import tqdm
from pathlib import Path
import requests
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from metAIsploit_assistant.types import BASE_MODELS, HackerModel


def get_default_model():
    default_model = BASE_MODELS.SNOOZY

    print("Making default Directories")
    Path(default_model.file_location).parent.mkdir(parents=True, exist_ok=True)

    print("Requesting download for the model")
    response = requests.get(default_model.url, stream=True)
    print("Saving response to file")
    with open(default_model.file_location, "wb") as fi:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            if chunk:
                fi.write(chunk)


def run_demo_from_prompt(hacker_model: HackerModel, prompt_question: str) -> str:
    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]

    # If you want to use a custom model add the backend parameter
    # Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
    llm = GPT4All(
        model=hacker_model.file_location,
        backend="gptj",
        callbacks=callbacks,
        verbose=True,
    )

    template = """Question: {question}

    Answer: Let's think step by step."""

    print(prompt_question)

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    llm_chain.run(prompt_question)


def main():
    get_default_model()
    run_demo_from_prompt(
        BASE_MODELS.SNOOZY,
        "Write a metasploit module for log4j vulnerability",
    )


def poetry_run_prompt_demo():
    run_demo_from_prompt(
        BASE_MODELS.SNOOZY,
        "Write a metasploit module for log4j vulnerability",
    )
