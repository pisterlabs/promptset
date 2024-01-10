from langchain.tools import BaseTool

from textcraft.vectors.pinecone_store import similarity_search


class SimilaritySearchTool(BaseTool):
    name = "向量存储工具"
    description = "文档存储pinecone"

    def _run(self, text: str, run_manager=None) -> str:
        return self.run_for_similarity_search(text)

    async def _arun(
        self,
        text: str,
        run_manager=None,
    ) -> str:
        pass

    def run_for_similarity_search(self, text):
        return similarity_search(text)
