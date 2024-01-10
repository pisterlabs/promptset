from langchain.tools import tool


class FileTools:
    @tool("Write File with content")
    def write_file(data):
        """write data to a file input to this tool is as follows file.txt | DATA_TO_WRITE"""
        try:
            # clean leading and trailing spaces
            path, content = data.strip().split("|")
            path = path.strip()
            content = content.strip().replace(" \\", "")
            with open(f"./crew_outputs/{path}", "w") as f:
                f.write(content)
        except Exception:
            return "Error with the input format for the tool."
