import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from .extract import Extractor, File
from .code import CodeExtractor, Code
from .utils import announce

__all__ = ["Memory"]

session_memory_path = "/tmp/session.csv"


class Work:
    def __init__(self, path: str, diff: str):
        """Initializer requires a file path and a diff string."""
        self.path = path
        self.diff = diff

    def concat(self):
        return f"Path: {self.path}\n Diff: {self.diff}"


class Memory:
    def __init__(self, extractor: Extractor, codeExtractor: CodeExtractor):
        """This class represents the memory of the system, storing completed work and embeddings."""
        self.extractor = extractor
        self.codeExtractor = codeExtractor
        self.completed_work = []
        self.embed()

    def embed(self):
        announce("Embedding code...")

        embeddings = []

        def _embed_code(code):
            embeddings.append(self._code_embedding(code))

        self.codeExtractor.extract(_embed_code)

        df = pd.DataFrame(
            embeddings, columns=["file_path", "language", "content", "embedding"]
        )
        df.to_csv(session_memory_path)

        announce("Done embedding code.")

    def add_work(self, file: File):
        work = Work(file.path, file.diff())
        self.completed_work.append(work)

        df = pd.read_csv(session_memory_path)
        embedding = self._file_embedding(file)
        df.at[file.path, "embedding"] = embedding["embedding"]
        df.to_csv(session_memory_path)

    def file_context(self, file: File):
        relevant_files = self._relevant_files(file)
        relevant_files_concat = "\n".join([file.concat() for file in relevant_files])
        relevant_files_message = {
            "role": "system",
            "content": f"Relevant files:\n{relevant_files_concat}",
        }

        completed_work_concat = "\n".join(
            [work.concat() for work in self.completed_work]
        )
        completed_work_message = {
            "role": "system",
            "content": f"Completed work:\n{completed_work_concat}",
        }

        return [relevant_files_message, completed_work_message]

    def code_context(self, file: File):
        relevant_code = self._relevant_code(file)
        relevant_code_concat = "\n".join([code.vect() for code in relevant_code])

        return relevant_code_concat

    def goal_code_context(self, goal: str):
        goal_code = self._goal_code(goal)
        goal_code_concat = "\n".join([code.vect() for code in goal_code])

        return goal_code_concat

    def _embedding(self, value):
        embedding = get_embedding(
            value, engine="text-embedding-ada-002", max_tokens=8000
        )

        return {"value": value, "embedding": embedding}

    def _file_embedding(self, file: File):
        embedding = get_embedding(
            file.content, engine="text-embedding-ada-002", max_tokens=8000
        )

        return {"path": file.path, "name": file.name, "embedding": embedding}

    def _code_embedding(self, code: Code):
        embedding = get_embedding(
            code.vect(), engine="text-embedding-ada-002", max_tokens=8000
        )

        return {
            "file_path": code.file_path,
            "language": code.language,
            "content": code.content,
            "embedding": embedding,
        }

    def _relevant_files(self, file: File):
        embedding = self._file_embedding(file)["embedding"]
        df = pd.read_csv(session_memory_path)
        df["embedding"] = df.embedding.apply(eval).apply(np.array)
        df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))

        paths = df.sort_values(by="similarity", ascending=False).head(3).path.tolist()

        files = []

        def _add_file(path):
            files.append(path)

        for path in paths:
            if isinstance(path, str):
                Extractor(path).extract(_add_file)

        return files

    def _relevant_code(self, file: File) -> list:
        embedding = self._file_embedding(file)["embedding"]
        df = pd.read_csv(session_memory_path)
        df = df[df.file_path != file.path]
        df["embedding"] = df.embedding.apply(eval).apply(np.array)
        df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))

        result = (
            df.sort_values(by="similarity", ascending=False).head(20).to_dict('records')
        )

        return [
            Code(res["file_path"], res["language"], res["content"]) for res in result
        ]

    def _goal_code(self, goal: str) -> list:
        embedding = self._embedding(f"Goal: {goal}")["embedding"]
        df = pd.read_csv(session_memory_path)
        df["embedding"] = df.embedding.apply(eval).apply(np.array)
        df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))

        result = (
            df.sort_values(by="similarity", ascending=False).head(20).to_dict('records')
        )

        return [
            Code(res["file_path"], res["language"], res["content"]) for res in result
        ]

