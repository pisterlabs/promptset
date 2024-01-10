from __future__ import annotations

import io
import json
import logging
import pathlib
import zipfile
from typing import Any, Final

import requests
from langchain.docstore.document import Document
from langchain.document_loaders import MathpixPDFLoader

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
            processed_file_format,  # type: ignore
            max_wait_time_seconds,
            should_clean_pdf,
            **kwargs,
        )

    @property
    def data(self) -> dict:
        conversion_formats = {f: True for f in self.processed_file_format if f != "mmd"}
        options = {
            "conversion_formats": conversion_formats,
            **self.other_request_parameters,
        }
        return {"options_json": json.dumps(options)}

    def load(self) -> dict[str, Document | str] | dict[str, str]:  # type: ignore
        pdf_id = self.send_pdf()
        contents = self.get_processed_pdf(pdf_id)
        if self.should_clean_pdf:
            if "mmd" not in contents:
                logger.warning(
                    "As 'should_clean_pdf' only supports mmd format, the cleaning process was skipped.  Please add 'mmd' in the 'processed_file_format' argument if you want to set 'True' for 'should_clean_pdf'."
                )
            else:
                contents["mmd"] = self.clean_pdf(contents["mmd"])
        if self.output_langchain_document:
            if "mmd" not in contents:
                logger.warning(
                    "As mmd format is not included in the output, the output format was not converted to Document.  Please add 'mmd' in the 'processed_file_format' argument if you want to set 'True' for 'output_langchain_document'."
                )
            else:
                metadata = {"source": self.source, "file_path": self.source}
                contents["mmd"] = Document(page_content=contents["mmd"], metadata=metadata)  # type: ignore
        return contents

    def get_processed_pdf(self, pdf_id: str) -> dict[str, str]:  # type: ignore
        self.wait_for_processing(pdf_id)
        responses = dict()
        for conversion_format in self.processed_file_format:
            url = f"{self.url}/{pdf_id}.{conversion_format}"
            response = requests.get(url, headers=self.headers)
            if conversion_format == "tex.zip":
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall(self.output_path_for_tex)
                    responses["tex.zip"] = str(self.output_path_for_tex)
            else:
                responses[conversion_format] = response.content.decode("utf-8")
        return responses
