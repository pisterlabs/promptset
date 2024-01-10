from abc import abstractmethod
import os, json, tiktoken
from podcastSummaryPlugins.abstractPluginDefinitions.abstractStorySummaryPlugin import (
    AbstractStorySummaryPlugin,
)
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv


class BaseSummaryPlugin(AbstractStorySummaryPlugin):
    def __init__(self):
        currentFile = os.path.realpath(__file__)
        currentDirectory = os.path.dirname(currentFile)
        load_dotenv(os.path.join(currentDirectory, ".env.summary"))

    @abstractmethod
    def summarizeText(self, story):
        pass

    @abstractmethod
    def identify(self) -> str:
        pass

    def writeToDisk(
        self, story, summaryText, summaryTextDirName, summaryTextFileNameLambda
    ):
        url = story["link"]
        uniqueId = story["uniqueId"]
        rawTextFileName = summaryTextFileNameLambda(uniqueId, url)
        filePath = os.path.join(summaryTextDirName, rawTextFileName)
        os.makedirs(summaryTextDirName, exist_ok=True)
        with open(filePath, "w", encoding="utf-8") as file:
            json.dump(summaryText, file)
            file.flush()

    def doesOutputFileExist(
        self, story, summaryTextDirName, summaryTextFileNameLambda
    ) -> bool:
        url = story["link"]
        uniqueId = story["uniqueId"]
        rawTextFileName = summaryTextFileNameLambda(uniqueId, url)
        filePath = os.path.join(summaryTextDirName, rawTextFileName)
        if os.path.exists(filePath):
            print(
                "Summary text file already exists at filepath: "
                + filePath
                + ", skipping summarizing story"
            )
            return True
        return False

    def prepareForSummarization(self, texts):
        if (
            self.numberOfTokensFromString(texts)
            < (4096 - int(os.getenv("OPENAI_MAX_TOKENS_SUMMARY"))) - 265
        ):
            return [texts]

        chunkSize = int(os.getenv("CHUNK_SIZE"))
        textSplitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator=".\n", chunk_size=chunkSize, chunk_overlap=0  # no overlap
        )
        splitTexts = textSplitter.split_text(texts)
        if len(splitTexts) <= 2:
            textSplitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator="<p>", chunk_size=chunkSize, chunk_overlap=0  # no overlap
            )
            splitTexts = textSplitter.split_text(texts)
        if len(splitTexts) <= 2:
            textSplitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator=" ", chunk_size=chunkSize, chunk_overlap=0  # no overlap
            )
            splitTexts = textSplitter.split_text(texts)
        if len(splitTexts) <= 2:
            raise ValueError(
                "Text cannot be summarized, please check the text and the above separators and try again."
            )
        return splitTexts

    def numberOfTokensFromString(self, string: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        numTokens = len(encoding.encode(string))
        return numTokens
