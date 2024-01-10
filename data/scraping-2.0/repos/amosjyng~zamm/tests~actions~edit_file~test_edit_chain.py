from langchain.prompts.prompt import PromptTemplate
from langchain_contrib.llms.testing import FakeLLM

from zamm.actions.edit_file import FileOutputChain
from zamm.utils import temporary_file


def test_write_tabs_to_file():
    file_path = "tests/resources/tab_write.txt"
    contents = "\ta"
    with temporary_file(file_path):
        FileOutputChain(
            llm=FakeLLM(sequenced_responses=["\ta"]),
            prompt=PromptTemplate(
                input_variables=["file_path", "file_contents"],
                template="Dummy {file_path} with {file_contents}",
            ),
        )(
            {
                "file_path": file_path,
                "file_contents": contents,
            }
        )
        with open(file_path) as f:
            assert f.read() == contents
