import os
import shutil
import traceback
from abc import ABC, abstractclassmethod
from multiprocessing import Process, Queue

import dill

from engines.openai_whisper import OpenAIWhisper
from models.whisper_task import WhisperTask


class BaseHandler(ABC):
    def __init__(self) -> None:
        self.queue = Queue()
        self._launch_worker_process()

    def _launch_worker_process(self):
        self.whisper_process = Process(target=self._launch_worker_func)
        self.whisper_process.start()

    def _launch_worker_func(self):
        print("开始加载模型...")
        w = OpenAIWhisper()
        print("模型加载完毕，等待接收任务...")
        while True:
            task = self.queue.get()
            print(f"开始处理任务：{task.media_path}")
            try:
                self.process_task(task, w)
            except:
                print("处理任务失败，跳过...")
                traceback.print_exc()
                self._handle_error_file(task.media_path)

    @abstractclassmethod
    def process_task(self, task: WhisperTask, w: OpenAIWhisper):
        pass

    def _handle_error_file(self, file_path):
        error_folder = os.path.join(os.path.dirname(file_path), "error_files")
        if not os.path.exists(error_folder):
            os.mkdir(error_folder)
        shutil.move(file_path, error_folder)

    def add_task(self, task: WhisperTask):
        print(f"添加任务：{os.path.basename(task.media_path)}")
        if str(task.media_path).lower().split(".")[-1] in ["txt", "srt"]:
            print("  文件格式不合法，跳过!")
            return
        self.queue.put(task)

    def close(self):
        self.whisper_process.terminate()
        self.whisper_process.close()
        print("模型已关闭！")
