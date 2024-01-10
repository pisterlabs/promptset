import os
import json
from dataclasses import dataclass
from typing import List
import httpx
from dotenv import load_dotenv
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from llama_index import StorageContext, load_index_from_storage
from llama_index import VectorStoreIndex, ServiceContext
from data_pipelines.parser_transcribe import ParserTranscribe, get_video_urls
from custom_embedding import OpenAIEmbeddingProxy

load_dotenv()
PROXY = os.getenv("PROXY")
http_client = httpx.Client(proxies=PROXY)

@dataclass()
class IndexPipeline:
    """
    Запускает пайплайн получения индекса -
    от определения новых видео в YouTube до сохранения индекса.
    """
    path_to_save: str
    url_file_path: str
    json_video_info_path: str
    index_folder: str
    chunk_size: int = 200
    chunk_overlap: int = 50

    def _get_download_urls(self, channel_url) -> List[str]:
        """Определяет список видео для скачивания"""

        # Determine the set of downloaded videos
        downloaded_videos = set()
        if os.path.exists(self.url_file_path):
            with open(self.url_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    downloaded_videos.add(line.strip())

        # Determine the set of videos to download
        new_videos = list(set(get_video_urls(channel_url)) - downloaded_videos)

        # Add new videos to the file
        with open(self.url_file_path, "a", encoding="utf-8") as f:
            for video in new_videos:
                f.write(video + "\n")

        return new_videos

    def _transcribe_videos(self, new_videos: List[str]) -> None:
        """
        Скачивает и транскрибирует видео из списка self.new_videos.
        Текст и метаданные сохраняются в json-файл.
        """
        transcriber = ParserTranscribe(self.path_to_save, self.json_video_info_path)
        for i, url in enumerate(new_videos):
            print(f"Transcribe {i} video")
            transcriber.get_transcribe_video(url)

    def _get_index(self, new_videos: List[str]) -> None:
        """
        Функция должна работать следующим образом:

        1. Если есть сохраненный индекс в self.storage_index_path -
        загружает его.
        2. По списку new_videos находит документы в json и добавляет их в индекс.
        Или создает новый индекс, если self.storage_index_path не существует
        3. Сохраняет индекс
        """
        embed_model = OpenAIEmbeddingProxy(http_client=http_client)
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        service_context = ServiceContext.from_defaults(node_parser=node_parser,
                                                       embed_model=embed_model)
        # Загружаем индекс
        index_store_path = os.path.join(self.index_folder, "index_store.json")
        if os.path.exists(index_store_path):
            storage_context = StorageContext.from_defaults(persist_dir=self.index_folder)
            index = load_index_from_storage(storage_context, service_context=service_context)
        else:
            # Или создаем пустой
            index = VectorStoreIndex([], service_context=service_context)

        # Выбираем документы для добавления в индекс
        with open(self.json_video_info_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)
        to_download_data = []
        for i in new_videos:
            for j in data_json:
                if i == j["url"][0]:
                    to_download_data.append(j)

        # Формируем документы
        documents = [
            Document(
                text=data["text"][0],
                metadata={"url": data["url"][0], "title": data["title"][0]},
            )
            for data in to_download_data
        ]

        # Добавляем документы в индекс
        for i, doc in enumerate(documents):
            print(f"Add {i} video to index")
            index.insert(doc)

        # Сохраняем индекс
        index.storage_context.persist(self.index_folder)


    def run(self, channel_url: str, test: bool=False) -> None:
        """Запускает пайплайн получения индекса"""
        new_videos = self._get_download_urls(channel_url)
        if new_videos:
            if test:
                new_videos = new_videos[:1]
            self._transcribe_videos(new_videos)
            self._get_index(new_videos)
        else:
            print("No new videos to download")

if __name__ == "__main__":
    pipe = IndexPipeline(
        "../data/audio",
        "../data/urls_of_channel_videos.txt",
        "../data/video_info.json",
        "../data/index_storage_1024",
        chunk_size=1024
    )
    pipe.run("https://www.youtube.com/c/karpovcourses")
