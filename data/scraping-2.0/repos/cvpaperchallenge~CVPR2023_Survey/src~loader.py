from __future__ import annotations

import io
import json
import pathlib
import zipfile
from typing import Any

import requests
from langchain.docstore.document import Document
from langchain.document_loaders import MathpixPDFLoader


class CustomMathpixLoader(MathpixPDFLoader):
    """Loader for mathpix.

    NOTE: This class extends `MathpixPDFLoader` class implemented in
    langchain to support request paramters.

    """

    def __init__(
        self,
        file_path: str,
        output_path_for_tex: pathlib.Path,
        processed_file_format: list[str] = ["mmd", "tex.zip"],
        max_wait_time_seconds: int = 500,
        should_clean_pdf: bool = False,
        output_langchain_document: bool = True,
        other_request_parameters: dict = {},
        **kwargs: Any,
    ) -> None:
        self.output_langchain_document = output_langchain_document
        self.other_request_parameters = other_request_parameters
        self.output_path_for_tex = output_path_for_tex
        super().__init__(
            file_path,
            processed_file_format,
            max_wait_time_seconds,
            should_clean_pdf,
            **kwargs,
        )

    @property
    def data(self) -> dict:
        conversion_formats = {f: True for f in self.processed_file_format}
        options = {
            "conversion_formats": conversion_formats,
            **self.other_request_parameters,
        }
        return {"options_json": json.dumps(options)}

    def load(self) -> list[Document] | str:
        pdf_id = self.send_pdf()
        contents = self.get_processed_pdf(pdf_id)
        if self.should_clean_pdf:
            contents = self.clean_pdf(contents)
        if self.output_langchain_document:
            metadata = {"source": self.source, "file_path": self.source}
            output = [Document(page_content=contents, metadata=metadata)]
        else:
            output = contents
        return output

    def get_processed_pdf(self, pdf_id: str) -> dict[str, str]:
        self.wait_for_processing(pdf_id)
        responses = dict()
        for conversion_formats in self.processed_file_format:
            url = f"{self.url}/{pdf_id}.{conversion_formats}"
            response = requests.get(url, headers=self.headers)
            if conversion_formats == "tex.zip":
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall(self.output_path_for_tex)
                    responses["tex.zip"] = self.output_path_for_tex
            else:
                responses[conversion_formats] = response.content.decode("utf-8")
        return responses
