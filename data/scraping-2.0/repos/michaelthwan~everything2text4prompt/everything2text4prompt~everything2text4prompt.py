import openai

from .pdf_util import PDFUtil
# from .podcast_util import convert_podcast_transcript
from .youtube_util import YoutubeUtil


class Everything2Text4Prompt:
    def __init__(self, openai_api_key, is_azure=False):
        self.openai_api_key = openai_api_key
        self.is_azure = is_azure
        openai.api_key = self.openai_api_key

    def convert_text(self, medium, target_source) -> (str, bool, str):
        if medium == "youtube":
            return YoutubeUtil.get_youtube_data(target_source)
        # elif medium == "podcast":
        #     return convert_podcast_transcript(target_source)
        elif medium == "pdf":
            return PDFUtil.get_pdf_data(target_source)
        else:
            raise Exception("Unsupported medium")


if __name__ == "__main__":
    openai_api_key = ""
    converter = Everything2Text4Prompt(openai_api_key)

    medium = "youtube"
    target_source = "8S0FDjFBj8o"  # Default English
    # target_source = "lSTEhG021Jc"  # Default auto-generated English
    # target_source = "https://www.youtube.com/watch?v=lSTEhG021Jc&ab_channel=EddieGM"  # Test the handling if people input URL
    # target_source = "https://www.youtube.com/watch?v=29WGNfuxIxc&ab_channel=PanSci%E6%B3%9B%E7%A7%91%E5%AD%B8"  # Default Chinese
    # target_source = "https://www.youtube.com/watch?v=K0SZ9mdygTw&t=757s&ab_channel=MuLi"  # Subtitle not available
    # target_source = "https://www.youtube.com/watch?v=MfDlgRtmgpc&ab_channel=%E9%98%BF%E8%B1%ACAhJu" # yue-HK language testing
    # target_source = "a"  # Error

    # medium = "podcast"
    # Short english
    # Moment 108 - This Powerful Tool Can Change Your Life: Africa Brooke
    # target_source = "https://podcasts.google.com/feed/aHR0cHM6Ly9mZWVkcy5idXp6c3Byb3V0LmNvbS8xNzE3MDMucnNz/episode/NWQzYmJlZDktNzA1Mi00NzU5LThjODctMzljMmIxNmJjZDM3?sa=X&ved=0CAUQkfYCahcKEwig_fW00YH_AhUAAAAAHQAAAAAQLA"

    # Long Chinese
    # TODO: Not sure why it is not working after chunking
    # 通用人工智能离我们多远，大模型专家访谈 ｜S7E11 硅谷徐老师 x OnBoard！
    # target_source = "https://podcasts.google.com/feed/aHR0cHM6Ly9mZWVkcy5maXJlc2lkZS5mbS9ndWlndXphb3poaWRhby9yc3M/episode/YzIxOWI4ZjktNTZiZi00NGQ3LTg3NjctYWZiNTQzOWZjMTNk?sa=X&ved=0CAUQkfYCahcKEwjwp9icjv_-AhUAAAAAHQAAAAAQLA&hl=zh-TW"

    data, is_success, error_msg = converter.convert_text(medium, target_source)
    print(data.shorten_transcript)
    print(is_success, error_msg)
    # print(data.ts_transcript_list)
