import os
import shutil
from handlers.base_handler import BaseHandler

from engines.openai_whisper import OpenAIWhisper
from models.whisper_task import WhisperTask
import dill


class Chan4Handler(BaseHandler):
    def process_task(self, task: WhisperTask, w: OpenAIWhisper):
        if task.target_languages:
            language = w.detect_language(task.media_path)
            if language not in task.target_languages:
                print(f"检测出语言为[{language}]，不在目标语言[{task.target_languages}]中，跳过处理...")
                language_folder = os.path.join(os.path.dirname(task.media_path), language)
                if not os.path.exists(language_folder):
                    os.mkdir(language_folder)
                shutil.move(task.media_path, language_folder)
                return
        result = w.transcribe(task.media_path, task.verbose)
        print("任务处理完成...")
        if task.post_func_bytes:
            post_func = dill.loads(task.post_func_bytes)
            post_func(task.media_path, result)
