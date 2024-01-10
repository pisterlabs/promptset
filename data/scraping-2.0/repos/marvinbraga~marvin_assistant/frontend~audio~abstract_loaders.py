from typing import Iterable

from langchain.document_loaders import BlobLoader, Blob, FileSystemBlobLoader


class AbstractAudioLoader(BlobLoader):
    glob = None

    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def yield_blobs(self) -> Iterable[Blob]:
        if self.glob is None:
            assert "Você deve informar um valor para o atributo 'glob' de sua classe."

        loader = FileSystemBlobLoader(self.save_dir, glob=self.glob)
        for blob in loader.yield_blobs():
            yield blob
