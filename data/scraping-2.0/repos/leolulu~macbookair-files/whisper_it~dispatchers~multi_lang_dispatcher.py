import os

from engines.openai_whisper import OpenAIWhisper
from utils.file_util import change_ext


class MultiLangDispatcher:
    def __init__(self, folder_path) -> None:
        self.folder_path = os.path.abspath(folder_path)
        self.w = OpenAIWhisper()

    def get_files(self):
        self.files = []
        for i in os.listdir(self.folder_path):
            i = os.path.join(self.folder_path, i)
            self.files.append(i)

    def get_langs(self):
        for media_path in self.files:
            lang_result = self.w.detect_language_by_longer_material(media_path)
            with open(change_ext(media_path, ".txt", postfix="lang_result"), "w", encoding="utf-8") as f:
                f.write(str(lang_result))

    