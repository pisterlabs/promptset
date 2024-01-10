from langchain.tools import BaseTool

class NoteMakingTool(BaseTool):
    name = "NoteMaker"
    description = "Give this tool a message with your note when you need to make or edit an exisiting note. Your message should be in the following format: <title> | <your note>"  # noqa: E501

    def _run(self, new_text: str):
        file_name = new_text.split("|")[0].rstrip() 
        text_to_write = new_text.split("|")[1]
        with open(f"./alexandria/{file_name}.md", "a", encoding="utf-8") as f:
            f.write(text_to_write)
        return f"Created a note called {file_name}"
    
    def _arun(self):
        raise NotImplementedError("This tool does not support async")

class NoteReaderTool(BaseTool):
    name = "NoteReader"
    description = "use this tool when you need to read a note"

    def _run(self, file_name: str):
        with open(f"./alexandria/{file_name}.md", "r", encoding="utf-8") as f:
            text = f.read()
        return text
    
    def _arun(self):
        raise NotImplementedError("This tool does not support async")
