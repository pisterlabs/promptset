from langchain.agents import Tool
from langchain.tools.file_management import (
    CopyFileTool,
    DeleteFileTool,
    ListDirectoryTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)

from .scoped_file_tools_funcs import (
    MyScriptExecutionTool,
    edit_file_tool_factory,
    file_tool_factory,
)


def build_scoped_file_tools(root_dir: str) -> list[Tool]:
    MyCreateFileTool, MyFillFileTool = file_tool_factory()
    MyReadLineTool, MyLocateLineTool, MyEditLineTool = edit_file_tool_factory()

    read_one_file_tool = Tool(
        name="read_one_file",
        func=ReadFileTool(
            root_dir=root_dir,
        ).run,
        description="""
Useful when you want to get the contents inside a file in a specified file path.
You should enter the file path recognized by the file. If you can not find the file,
""",
    )

    read_directory_tree_tool = Tool(
        name="read_directory_tree",
        func=ListDirectoryTool(
            root_dir=root_dir,
        ).run,
        description="""
Useful when you need to know what files are contained in this project.
You should run this to record the file directory tree when you need to.
""",
    )

    create_file_tool = Tool(
        name="create_file",
        func=MyCreateFileTool(
            root_dir=root_dir,
        ).run,
        description="""
Useful when you want to create files.
You should run this to create the file right before writing to the file.
A use of the "write_file" tool should immediately follow this tool.
""",
    )

    fill_file_tool = Tool(
        name="write_file",
        func=MyFillFileTool(
            root_dir=root_dir,
        ).run,
        description="""
Useful when you want to fill in the contents in a file.
This tool should immediately follow a use of the "create_file" tool.
You should run this to write in a file, the file must be created first.
Don't include the file path, just include the file content.
Follow this example strictly.
""",
    )
    read_line_tool = Tool(
        name="read_line",
        func=MyReadLineTool(
            root_dir=root_dir,
        ).run,
        description="""
Useful when you want to edit the contents in a file.
You should run this FIRST before trying to edit the file.
You should locate the lines where you want to edit the file with this tool.
After this tool, you should always verify the line numbers with locate_line tool.
""",
    )
    locate_line_tool = Tool(
        name="locate_line",
        func=MyLocateLineTool(
            root_dir=root_dir,
        ).run,
        description="""
Useful when you want to verify the lines exists in a file.
This tool should immediately follow a use of the "read_line" tool.
You should run this with the input STRICTLY,
following the example format here: num1:num2,
with num1 being the starting line of edit happens
and num2 being the ending line of edit happens.
in a string, with num1 being the starting line of edit happens
and num2 being the ending line of edit happens.
,the file containing the lines must be read first.
After this tool, you should edit file with edit_line tool.
""",
    )

    edit_line_tool = Tool(
        name="edit_line",
        func=MyEditLineTool(
            root_dir=root_dir,
        ).run,
        description="""
Useful when you want to write specific contents in seleted lines in a file.
This tool should immediately follow a use of the "locate_line" tool.
You should run this to edit lines in a file,
the lines must be verified by locate_line_tool first.
""",
    )
    test_execution_tool = Tool(
        name="test_execution",
        func=MyScriptExecutionTool(
            root_dir=root_dir,
        ).run,
        description="""
Useful when you want to test a script you created or edited.
This tool should be used often to ensure correctness.
The input to this tool should be a string, most importantly,
with the specific file path that includes the script name.
For example, if you want to test a script "test_script.py",
under the directory "test", you should enter "test/test_script.py".
""",
    )

    delete_file_tool = DeleteFileTool(
        root_dir=root_dir,
    )

    return [
        read_one_file_tool,
        read_directory_tree_tool,
        # create_file_tool,
        # fill_file_tool,
        delete_file_tool,
        # FIXME fix these tools
        # read_line_tool,
        # locate_line_tool,
        # edit_line_tool,
        # test_execution_tool,
    ]
