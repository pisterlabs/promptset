from langchain import HuggingFaceHub, LLMChain, PromptTemplate
import time
import os
from events import post_event

from typing import Optional

LLM_CHAIN: Optional[LLMChain] = None


def initialize_langchain(api_key: Optional[str], model_name="sangam101/hpc") -> bool:
    """
    Initialize the language chain for running inference

    Parameters
    ---
    model_name: The model name followed by repo id for the model stored in huggingface.
    api_key: The huggingface api key, if none is provided, api key is searched in environment variables

    Return
    ---
    False if api token isn't found, True otherwise

    """
    global LLM_CHAIN

    if api_key is None:
        api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

        if api_key is None:
            post_event("log", "Huggingface hub api key not found, try setting the environment variable HUGGINGFACEHUB_API_TOKEN")
            return False
        
        post_event("log", f"Using huggingfacehub api key {api_key}")

    hub_llm = HuggingFaceHub(
        repo_id=model_name,
        model_kwargs={"temperature": 1e-10},
        huggingfacehub_api_token=api_key,
        client=None,  # to solve intellisense error
    )

    template = (
        """{comment}"""  # passing just the comment, as the model is trained for that
    )
    prompt = PromptTemplate(template=template, input_variables=["comment"])

    # create prompt template > LLM chain
    LLM_CHAIN = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)


    return True


def get_variable_names_from_langchain(comments: list[str], **kwargs) -> Optional[dict[str, str]]:
    """
    Get the variable names for comments at once.
    Kwargs isn't used for now.

    Parameters
    ---
    comments: list of comment


    Return
    ---
    None if the llm chain is invalid.

    Dictionary containing the comment as key and variable name as value if the llm chain is valid.

    """
    global LLM_CHAIN

    if not isinstance(LLM_CHAIN, LLMChain) or len(comments) < 1:
        return None

    # generating a list of dictionary of comments
    comment_dictionaries: list[dict[str, str]] = []
    for comment in comments:
        # the input variable name is comment as set above
        # this should be changed on changing the input variable name
        comment_dictionaries.append({"comment": comment})

    post_event("log", f"Starting inference on langchain for {comment_dictionaries}")
    start_time = time.time()
    outputs = LLM_CHAIN.generate(comment_dictionaries)
    end_time = time.time()
    post_event("log", f"Finished inference on langchain for {comment_dictionaries}")
    post_event("log", f"\tThe inference took {end_time - start_time} seconds.\n")

    comment_variable_dict: dict[str, str] = {}

    for i, output in enumerate(outputs.generations):
        input_comment = comments[i]
        output_variable = output[0].text

        comment_variable_dict[input_comment] = output_variable

    return comment_variable_dict


if __name__ == "__main__":
    # the huggingface model name
    model_name = "sangam101/hpc"

    # the input comment
    comments = [
        "Determining the longest running execution.",
        "calculate the area of circle",
        "get the most accessed websites",
        "keeping track of maximum input size",
    ]

    initialize_langchain(api_key=None, model_name=model_name)

    langchain_inference_start_time = time.time()

    outputs = get_variable_names_from_langchain(comments=comments)

    langchain_inference_end_time = time.time()

    print(
        f"Total time taken for inference = {langchain_inference_end_time - langchain_inference_start_time} seconds"
    )

    if outputs is not None:
        for comment, variable_name in outputs.items():
            print(f"Comment: {comment}\n\t Variable Name: {variable_name}")
