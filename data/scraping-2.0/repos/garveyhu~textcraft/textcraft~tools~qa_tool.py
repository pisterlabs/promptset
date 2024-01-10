from langchain.tools import BaseTool

from textcraft.vectors.pinecone_qa import vector_qa


class QATool(BaseTool):
    name = "向量数据库问答工具"
    description = "数据库QA"

    def _run(self, text: str, run_manager=None) -> str:
        return self.run_for_title(text)

    async def _arun(
        self,
        text: str,
        run_manager=None,
    ) -> str:
        pass

    def run_for_title(self, text):
        return vector_qa(text)
