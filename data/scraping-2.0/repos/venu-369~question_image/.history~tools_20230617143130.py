from langchain.tools import BaseTool

class ImageCaptionTool(BaseTool):
    name = None
    description = None

    def _run(self,img_path):
        pass
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")