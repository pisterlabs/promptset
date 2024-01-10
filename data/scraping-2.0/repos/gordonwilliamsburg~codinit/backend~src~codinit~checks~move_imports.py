"""
The purpose is to move imports that happen in the code body to the top, where all imports are expected
Example:
    Input:
        llm_chain = langchain.LLMChain(llm=llms.OpenAI())
        agent = langchain.agents.Agent(llm_chain=llm_chain)
    Output:
        from langchain import agents
        from langchain import LLMChain
        llm_chain = LLMChain(llm=llms.OpenAI())
        agent = agents.Agent(llm_chain=llm_chain)
"""
from typing import Set

import libcst as cst


class LibraryUsageCollector(cst.CSTVisitor):
    """
    A CSTVisitor subclass that collects usages of a specific library in the code.
    """

    def __init__(self, library_name: str):
        """
        Initialize the collector.

        Args:
        - library_name (str): The name of the library to collect usages for.
        """
        self.library_name = library_name
        self.library_usages: Set[str] = set()

    def visit_Import(self, node: cst.Import) -> bool:
        """Skip visiting children of the Import node."""
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        """Skip visiting children of the ImportFrom node."""
        return False

    def visit_Attribute(self, node: cst.Attribute) -> None:
        """Collect attribute usages that belong to the specified library."""
        if isinstance(node.value, cst.Name) and node.value.value == self.library_name:
            self.library_usages.add(node.attr.value)


class RemoveLibraryPrefix(cst.CSTTransformer):
    """
    A CSTTransformer subclass that removes the prefix of a specific library from the code.
    """

    def __init__(self, library_name: str):
        """
        Initialize the transformer.

        Args:
        - library_name (str): The name of the library whose prefix should be removed.
        """
        self.library_name = library_name

    def visit_Import(self, node: cst.Import) -> bool:
        """Skip visiting children of the Import node."""
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        """Skip visiting children of the ImportFrom node."""
        return False

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.BaseExpression:
        """Replace library attribute usage with just the attribute, removing the library prefix."""
        if (
            isinstance(updated_node.value, cst.Name)
            and updated_node.value.value == self.library_name
        ):
            return updated_node.attr
        return updated_node


def refactor_code(code: str, library_name: str) -> str:
    """
    Refactor the code to remove the prefix of a specific library and add necessary import statements.

    Args:
    - code (str): The original code to refactor.
    - library_name (str): The name of the library to refactor.

    Returns:
    - str: The refactored code.
    """
    module = cst.parse_module(code)

    collector = LibraryUsageCollector(library_name)
    module.visit(collector)

    # Get all unique library usages
    library_usages = collector.library_usages

    # Add missing imports
    new_imports = [
        cst.ImportFrom(
            module=cst.Name(library_name), names=[cst.ImportAlias(name=cst.Name(usage))]
        )
        for usage in library_usages
    ]

    # Insert the new imports at the top
    new_body = list(module.body)
    new_body[0:0] = [cst.SimpleStatementLine(body=[imp]) for imp in new_imports]

    refactored_module = module.with_changes(body=new_body).visit(
        RemoveLibraryPrefix(library_name)
    )

    return refactored_module.code


if __name__ == "__main__":
    original_code = "src/codinit/checks/sample_code/sample_code.py"
    processed_code = "src/codinit/checks/sample_code/sample_corrected_code.py"
    with open(original_code, "r") as file:
        code = file.read()
    # refactor_code(code)
    print(refactor_code(code, library_name="langchain"))
