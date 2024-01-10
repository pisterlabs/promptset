import csv
import os

from datetime import date

from langchain.tools import StructuredTool
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.agents.agent_toolkits import FileManagementToolkit
from tempfile import TemporaryDirectory

current_directory = os.getcwd()

def create_csv_file(args: str):
    input = args.split(",")
    csv_name = input.pop(0)

    with open(csv_name, "w", newline="") as file:
        if input[0] != '[]':
            writer = csv.DictWriter(file, fieldnames=input, dialect=csv.excel, delimiter=",")
            writer.writeheader()
        response = (f"CSV file '{csv_name}' has been created successfully.")
        return response


def add_row_to_csv(args: str) -> str:
    input = args.split(",")
    csv_name = input.pop(0)

    with open(csv_name, "a", newline="") as file:
        writer = csv.writer(file, dialect=csv.excel, delimiter=",")
        writer.writerow(input)
        response = (f"A row has been added to the CSV file '{csv_name}' successfully.")
        return response

def delete_csv_file(
        csv_name:str, 
        ) -> str:
    if os.path.exists(csv_name):
        os.remove(csv_name)
        response = (f"CSV file '{csv_name}' has been deleted successfully.")
        return response
    else:
        response = (f"CSV file '{csv_name}' does not exist.")
        return response

def charge_csv(
        csv_name: str
        ) -> str:
            qa = charge_csv(csv_name=csv_name)
            return ""

def get_current_date() -> str:
    today = date.today()
    now = today.strftime("%B %d, %Y")

    return now

tool_get_current_date = StructuredTool.from_function(
        get_current_date,
        description="Get current date"
        )

tool_create_csv_file = StructuredTool.from_function(
        create_csv_file,
        description="Create a csv file"
        )
tool_add_row_to_csv = StructuredTool.from_function(
        add_row_to_csv,
        description="Edit a csv file"
        )
tool_delete_csv_file = StructuredTool.from_function(
        delete_csv_file,
        description="Delete a csv file"
        )
tool_charge_csv_file = StructuredTool.from_function(
        charge_csv,
        description="Charge csv"
        )

tools_default_file_management = FileManagementToolkit(
        root_dir=str(current_directory),
        selected_tools=["read_file", "list_directory", "file_search"]
        ).get_tools()
