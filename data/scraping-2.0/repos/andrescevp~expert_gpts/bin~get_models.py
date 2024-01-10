import click
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate


@click.command()
def load_docs():
    llm = HuggingFacePipeline.from_model_id(
        model_id="tiiuae/falcon-7b-instruct",
        task="text-generation",
        model_kwargs={
            "temperature": 0,
            "max_tokens": 2048,
            "device_map": "auto",
            "trust_remote_code": True,
        },
        pipeline_kwargs={"max_tokens": 2048, "trust_remote_code": True},
    )

    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm

    question = "What is electroencephalography?"

    print(chain.invoke({"question": question}))


if __name__ == "__main__":
    load_docs()
