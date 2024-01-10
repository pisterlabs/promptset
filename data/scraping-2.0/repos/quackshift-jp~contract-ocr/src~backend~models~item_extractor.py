from abc import abstractmethod, ABC
from openai import OpenAI


class ItemExtractor(ABC):
    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client

    @abstractmethod
    def extract_items(self) -> dict[str, dict[str, any]]:
        """OCRテキストから、特定の情報を辞書形式で書き出す
        args:
            image_path str: 読み込み対象となる画像が存在しているパス
        return:
            dict[str, dict[str, str]]: 契約書内の項目とその項目の内容
            ex:
                {
                    "content": {
                        "物件名": "物件A",
                        "賃料": 100,
                        "契約日": "2023年1月1日",
                    }
                }
        """
        raise NotImplementedError()
