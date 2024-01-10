from halo import Halo

from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser

from ..llm import get_llm
from config import Models


# def _parse_output(content: str) -> str:
#     return content.split(",")


prompt = """
Determine the paths to all the files imported into the files below from the project root directory in the form of ./path/to/file with the correct file extension. Return the result as a comma separated list of file paths. Don't return anything else, just the file paths.

{file}
"""


@Halo(text="Loading relative files", spinner="dots")
def get_imported_file_paths(file: str):
    chain = LLMChain(
        llm=get_llm(Models.INTERPRETATION_MODEL),
        prompt=PromptTemplate.from_template(prompt),
        output_parser=CommaSeparatedListOutputParser(),
    )

    paths = chain.predict(file=file)

    return paths
