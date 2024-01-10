from langchain.tools import BaseTool

class ImageCaptionTool(BaseTool):
    name = "Image Captioner"
    description = "Use this tool when given the path to an image that you would like to be described."\
                  "It will return the best caption describing the image. "

    def _run(self,img_path):
        pass
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    

class ObjectDetectionTool(BaseTool):
    name = None
    description = None