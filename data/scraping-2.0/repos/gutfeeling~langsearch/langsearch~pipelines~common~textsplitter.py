import logging

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langsearch.pipelines.base import BasePipeline
from langsearch.utils import openai_length_function

logger = logging.getLogger(__name__)


class TextSplitterPipeline(BasePipeline):
    INPUTS = {
        "text": "text",
        "url": "url"
    }
    SECTIONS = "text_splitter_pipeline_sections"
    TEXT_SPLITTER_CLASS = RecursiveCharacterTextSplitter
    TEXT_SPLITTER_CLASS_PARAMS = {
        "chunk_size": 512,
        "chunk_overlap": 0,
        "length_function": openai_length_function
    }
    SIZE_CUTOFF = 40

    def __init__(self, text_splitter_class, text_splitter_class_params, size_cutoff, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_splitter = text_splitter_class(**text_splitter_class_params)
        self.size_cutoff = size_cutoff

    @classmethod
    def from_crawler(cls, crawler):
        text_splitter_class = cls.get_setting_from_partial_key(crawler.settings, "TEXT_SPLITTER_CLASS")
        if isinstance(text_splitter_class, str):
            text_splitter_class = cls.get_from_dotted(text_splitter_class)
        text_splitter_class_params = cls.get_setting_from_partial_key(crawler.settings, "TEXT_SPLITTER_CLASS_PARAMS")
        if isinstance(text_splitter_class_params, str):
            text_splitter_class_params = cls.get_params_from_file(text_splitter_class_params)
        size_cutoff = cls.get_setting_from_partial_key(crawler.settings, "SIZE_CUTOFF")
        return cls(text_splitter_class, text_splitter_class_params, size_cutoff)

    def apply(self, item, spider):
        if not hasattr(self, "text") or self.text is None:
            return item
        if not hasattr(self, "url"):
            return item
        # TODO: We calculate the token count twice here. We should probably improve the text splitter so that it outputs
        # the token count along with the sections.
        sections = self.text_splitter.split_text(self.text)
        sections = [section for section in sections if self.text_splitter._length_function(section) > self.size_cutoff]
        item[self.SECTIONS] = sections
        return item
