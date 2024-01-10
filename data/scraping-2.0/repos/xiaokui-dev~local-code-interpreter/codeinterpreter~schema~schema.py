from langchain.pydantic_v1 import BaseModel
from langchain.schema import HumanMessage, AIMessage
from loguru import logger


class File(BaseModel):
    name: str
    content: bytes

    @classmethod
    def from_path(cls, path: str):
        if not path.startswith("/"):
            path = f"./{path}"
        with open(path, "rb") as f:
            path = path.split("/")[-1]
            return cls(name=path, content=f.read())


class UserRequest(HumanMessage):
    file: File = None


class AIResponse(AIMessage):
    files: list[File] = []
    code_log: list[tuple[str, str]] = []

    def show(self):
        logger.info("AI: = {}", self.content)
        for file in self.files:
            print("File: ", file.name)
            file.show_image()
