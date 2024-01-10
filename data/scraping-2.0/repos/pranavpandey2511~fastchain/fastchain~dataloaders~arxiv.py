from base_loader import BaseDataloader
from langchain.document_loaders import ArxivLoader


class ArxivDataLoader(BaseDataloader):
    def __init__(self) -> None:
        super().__init__()

    def load_data(self):
        return super().load_data()

    def _verify_data(self):
        return super()._verify_data()
