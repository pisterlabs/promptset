from langchain.tools.base import BaseTool

class HelloTool(BaseTool):
    """Tool that generates a personalized hello message."""

    name = "HelloTool"
    description = (
        "A tool that generates a personalized hello message. "
        "Input should be a name string."
    )

    def _run(self, name: str = None) -> str:
        return f"Hello {name}!"

    async def _arun(self, name: str = None) -> str:
        """Use the HelloTool asynchronously."""
        return self._run(name)
