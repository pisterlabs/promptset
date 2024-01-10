from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter


class LoadFiles:
    def __init__(self, repo_path) -> None:
        self.repo_path = repo_path

    async def process_files(self):
        languages = [
            Language.PYTHON,
            Language.MARKDOWN,
            Language.HTML,
            Language.CPP,
            Language.GO,
            Language.JAVA,
            Language.JS,
            Language.PHP,
            Language.PROTO,
            Language.RST,
            Language.RUBY,
            Language.RUST,
            Language.SCALA,
            Language.SWIFT,
            Language.LATEX,
            Language.SOL,
        ]

        suffixes = [
            ".py",
            ".md",
            ".html",
            ".cpp",
            ".go",
            ".java",
            ".js",
            ".php",
            ".proto",
            ".rst",
            ".rb",
            ".rs",
            ".scala",
            ".swift",
            ".tex",
            ".sol",
        ]

        final_texts = []
        for language in languages:
            documents = await self.load_files(language, suffixes[languages.index(language)])
            texts = await self.splitting(documents, language)
            final_texts += texts

        return final_texts

    async def load_files(self, language, suffix):
        loader = GenericLoader.from_filesystem(
            self.repo_path + "/",
            glob="**/*",
            suffixes=[suffix],
            parser=LanguageParser(
                language=language,
                parser_threshold=500,
            ),
        )

        documents = loader.load()
        # print(len(documents))
        return documents

    async def splitting(self, documents, language):
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=2000, chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        return texts
