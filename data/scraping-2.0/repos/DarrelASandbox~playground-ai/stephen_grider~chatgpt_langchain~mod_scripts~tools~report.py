"""
This module contains a utility for writing HTML content to a file. 
It leverages the StructuredTool class from the langchain.tools module 
to create a tool that can be integrated into langchain workflows. 

The primary function 'write_report' is designed to write HTML content to a specified file,
which is particularly useful for generating and saving reports in HTML format. 

The 'WriteReportArgsSchema' class is a Pydantic model that defines the schema for 
the arguments required by the 'write_report' function. 
This ensures structured and validated input when the function is
used as a tool in the langchain framework.

Function:
- write_report: Writes HTML content to a file on disk.

Class:
- WriteReportArgsSchema: Pydantic model for validating arguments passed to 
                         the 'write_report' function.

Usage:
The 'write_report_tool' can be used in any system where there is a need to generate
and save HTML reports, especially when used in conjunction with other langchain tools and agents.
"""

from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel


def write_report(filename, html):
    """
    Writes HTML content to a file.

    This function takes a filename and HTML content as input and
    writes the HTML content to the specified file.
    It's designed to be used for generating and saving reports or
    any other HTML formatted documents.

    Args:
        filename (str): The name of the file where the HTML content will be written.
        html (str): The HTML content to be written to the file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)


class WriteReportArgsSchema(BaseModel):
    """
    A Pydantic model that defines the schema for arguments to the 'write_report' function.

    This class ensures that the inputs to the 'write_report' function are structured and validated,
    which is particularly useful when integrating this function as a tool in langchain workflows.

    Attributes:
        filename (str): The name of the file to write to.
        html (str): The HTML content to be written to the file.
    """

    filename: str
    html: str


# Define a StructuredTool for the 'write_report' function
write_report_tool = StructuredTool.from_function(
    name="write_report",
    description="Write an HTML file to disk. Use this tool whenever someone asks for a report.",
    func=write_report,
    args_schema=WriteReportArgsSchema,
)
