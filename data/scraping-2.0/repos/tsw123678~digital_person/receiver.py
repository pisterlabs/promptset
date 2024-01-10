# -*- coding = utf-8 -*-
# @Date: 2023/9/25
# @Time: 21:51
# @Author:tsw
# @File：receiver.py
# @Software: PyCharm
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from PySide2.QtMultimedia import QMediaContent, QMediaPlayer
from PySide2.QtWidgets import QApplication, QMainWindow, QVBoxLayout, \
    QSizePolicy
from PySide2.QtCore import QTimer, QUrl, QCoreApplication
from PySide2.QtUiTools import QUiLoader
from VALL.make_Audio_class import generate_and_save_audio
import time
from OpenGL.GLUT import *
from PySide2.QtMultimediaWidgets import QVideoWidget
from utils.toAVI import convert_mp4_to_avi
import openai
from PySide2.QtGui import QPixmap


def getpath():
    source_dir = './temp/final_output'
    target_dir = './temp/final_output'
    convert_mp4_to_avi(source_dir, target_dir, delete_original=True)

    global latest_file_path
    directory_path = './temp/final_output'
    files = os.listdir(directory_path)
    timestamped_files = [file for file in files if '_' in file and '.' in file]
    if not timestamped_files:
        print("没有找到符合条件的文件")
    else:
        sorted_files = sorted(timestamped_files, key=lambda x: os.path.getmtime(os.path.join(directory_path, x)),
                              reverse=True)
        latest_file_name = sorted_files[0]
        latest_file_path = os.path.join(directory_path, latest_file_name)
        latest_file_path = latest_file_path.replace("\\", "/")
        print("最新文件的完整相对路径是:", latest_file_path)

    return latest_file_path


# 主窗口
class Main(QMainWindow):

    def __init__(self, ui_file):
        super().__init__()

        self.setWindowTitle("Digital Person : Receiver")

        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        self.setCentralWidget(self.ui)

        # 7.播放视频
        self.openGLWidget = self.ui.openGLWidget

        layout = QVBoxLayout()
        self.vw = QVideoWidget()
        layout.addWidget(self.vw)
        self.openGLWidget.setLayout(layout)

        layout.addWidget(self.vw, 1)
        self.vw.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.player = None

        # 监测
        # self.monitor_folder = "\\\\192.168.137.19\\share"
        self.monitor_folder = "D:\project_test"
        self.target_files = {"语义解码.txt": self.handle_semantic, "信道解码.txt": self.handle_channel}
        self.detected_files = set()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_folder)
        self.timer.start(2000)

        self.ui.text.setWordWrap(True)

    def load_img_voice(self):
        image_path = './temp/img/voice.png'
        pixmap = QPixmap(image_path)

        label_width = self.ui.voice.width()
        label_height = self.ui.voice.height()

        pixmap = pixmap.scaledToWidth(label_width)
        pixmap = pixmap.scaledToHeight(label_height)

        self.ui.voice.setPixmap(pixmap)

    def handle_semantic(self):
        print("语义解码已完成")
        full_path = os.path.join(self.monitor_folder, "语义解码.txt")
        os.remove(full_path)

        time.sleep(1)
        self.ui.state.setText("Semantic decode is completed.")
        QCoreApplication.processEvents()

        # 显示原图片
        self.load_img()
        QCoreApplication.processEvents()

        # 音色图片
        self.load_img_voice()
        QCoreApplication.processEvents()

        # 生成音频
        start_time = time.time()
        self.geneAudio()
        QCoreApplication.processEvents()

        # 合成视频
        self.make_final_video()
        end_time = time.time()
        self.rebuild_time = "{:.2f}".format(end_time - start_time)
        QCoreApplication.processEvents()

        # 显示信息
        text = f"\nDuration\n\nDecoding：3.12s\n\nRebuild：{self.rebuild_time}s"
        self.ui.info.setText(text)
        QCoreApplication.processEvents()

        self.ui.state.setText("Video rebuild is completed.Autoplay immediately!")
        time.sleep(1)
        QCoreApplication.processEvents()
        self.playVideo()

    def handle_channel(self):
        print("信道解码已完成")
        self.ui.state.setText("Channel decode is completed.")
        full_path = os.path.join(self.monitor_folder, "信道解码.txt")
        os.remove(full_path)

    def check_folder(self):
        for target_file, handler in self.target_files.items():
            if target_file in os.listdir(self.monitor_folder) and target_file not in self.detected_files:
                handler()
                self.detected_files.add(target_file)
                if len(self.detected_files) == len(self.target_files):
                    self.timer.stop()
                else:
                    self.detected_files.clear()

    def load_img(self):
        image_path = './temp/img/tsw.png'
        pixmap = QPixmap(image_path)

        label_width = self.ui.img.width()
        label_height = self.ui.img.height()

        pixmap = pixmap.scaledToWidth(label_width)
        pixmap = pixmap.scaledToHeight(label_height)

        self.ui.img.setPixmap(pixmap)

    def playVideo(self):

        # 播放
        try:
            self.player = QMediaPlayer()
            self.player.setVideoOutput(self.vw)
            video_file_path = getpath()
            media_content = QMediaContent(QUrl.fromLocalFile(video_file_path))
            self.player.setMedia(media_content)
            self.player.play()
            self.vw.show()
        except Exception as e:
            print("发生错误：", str(e))

    def make_final_video(self):
        self.ui.state.setText("Video is currently being reconstructed!Please wait.")
        QCoreApplication.processEvents()
        audio_directory = './temp/generateAudio'
        pattern = os.path.join(audio_directory, 'regeneAudio_*.wav')
        audio_files = glob.glob(pattern)
        audio_files.sort(key=os.path.getmtime, reverse=True)

        if audio_files:
            latest_audio_relative_path = audio_files[0]

            print("最新音频文件的相对路径:", latest_audio_relative_path)
        else:
            print("未找到音频文件")

        image = './temp/img/tsw.png'
        dir = './temp/final_output'

        os.system(
            "python ./SadTalker/inference.py --driven_audio %s --source_image %s --enhancer gfpgan --result_dir %s --size 256 --preprocess crop --pose_style 2 --expression_scale 2" % (
                latest_audio_relative_path, image, dir))

    def geneAudio(self):
        # 硬件
        # directory_path = "\\\\192.168.137.19\\share\\rec"
        # txt_files = os.listdir(directory_path)
        #
        # latest_txt_file = None
        # latest_timestamp = 0
        #
        # txt_files = [f for f in os.listdir(directory_path) if f.endswith('_output.txt')]
        # txt_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory_path, x)), reverse=True)
        # newest_txt_file = txt_files[0]
        #
        # latest_txt_file = os.path.join(directory_path, newest_txt_file)

        # 本地test
        latest_txt_file = './temp/generateTXT/txt/text_20231105195928.txt'

        if latest_txt_file:
            with open(latest_txt_file, "r", encoding="gbk") as file:
                content = file.read()

            # gpt精简
            openai.api_base = "https://api.closeai-proxy.xyz/v1"
            openai.api_key = "sk-HAG8bxQJ8pLF4W6UTZUjz679G6Hqf4FKqGPFvUYRDKpOe4Bb"

            os.environ["HTTP_PROXY"] = "127.0.0.1:33210"
            os.environ["HTTPS_PROXY"] = "127.0.0.1:33210"

            q = "用更少的词精简下面这句话但不要改变语义：{}".format(content)

            rsp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": q}
                ]
            )

            content = rsp['choices'][0]['message']['content']
            print("已完成精简，精简后内容为{}".format(content))

            timestamp = time.strftime("%Y%m%d%H%M%S")
            audio_filename = f'./temp/generateAudio/regeneAudio_{timestamp}.wav'

            self.ui.text.setText(content)
            QCoreApplication.processEvents()
            generate_and_save_audio(content, audio_filename)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = Main("./UI/receiver.ui")
    player.resize(1181, 674)
    player.show()
    sys.exit(app.exec_())
