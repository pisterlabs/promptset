from langchain.agents import Tool

class CodeEditorTooling():
    def __init__(self, code_editor):
        self.code_editor = code_editor

    def build_add_codes_tool(self):
        return Tool(
            name="CodeEditorAddCodes",
            func=self._add_codes_tool,
            description="""Use to add new lines of code. First line of input is line number and second line or later are new codes to insert.
            
Example that appends code at line 2:

Source Code: contract GeneratedContract {{
}}

Action: CodeEditorAddCodes
Action Input:
2
    uint256 x = 1;
    uint256 y = 2;

Observation: contract GeneratedContract {{
    uint256 x = 1;
    uint256 y = 2;
}}
""",
        )

    def build_change_code_tool(self):
        return Tool(
            name="CodeEditorChangeCodeLine",
            func=self._change_code_tool,
            description="""Use to modify an existing line of code. First line of input is line number and second line is new line of code to insert.

Example that modifies line 3:

Source Code: contract GeneratedContract {{
    uint256 x = 1;
    uint256 y = 2;
}}

Action: CodeEditorChangeCodeLine
Action Input:
3
uint256 x = 1;
uint256 z = 3;

Observation: contract GeneratedContract {{
    uint256 x = 1;
    uint256 z = 3;
}}
""",
        )

    def build_delete_codes_tool(self):
        return Tool(
            name="CodeEditorDeleteLine",
            func=self._delete_codes_tool,
            description="""Use to delete lines of code. Input has only the first line. The first line is indices of lines to delete, separated by comma. 

Example, to delete lines 1 and 3 of the source code.

Source Code: contract GeneratedContract {{
    uint256 x = 1;
    uint256 y = 2;
}}

Action: CodeEditorDeleteLine
Action Input:
1, 3

Observation:
    uint256 x = 1;
}}
""",
        )

    def _add_codes_tool(self, query: str) -> str:
        query_lines = query.splitlines()
        from_index = int(query_lines[0]) - 1
        new_codes = query_lines[1:]
        self.code_editor.add_codes(from_index, new_codes)

        return self.code_editor.get_code()

    def _change_code_tool(self, query: str) -> str:
        query_lines = query.splitlines()
        index = int(query_lines[0]) - 1
        new_code = query_lines[1]
        self.code_editor.change_code(index, new_code)

        return self.code_editor.get_code()

    def _delete_codes_tool(self, query: str) -> str:
        query_lines = query.splitlines()
        lines_to_delete = [int(x) for x in query_lines[0].split(",")]
        lines_to_delete.sort()
        lines_to_delete.reverse()

        for line in lines_to_delete:
            self.code_editor.delete_codes(line - 1)

        return self.code_editor.get_code()







