from pathlib import Path
from langchain_core.documents import Document
from . import BaseTool
from .. import config
import logging
logger = logging.getLogger('heymans')


class TopicsTool(BaseTool):
    """A more targeted form of search where a selection of prespecified topics
    is chosen and associated documents are added to the documentation object.
    """
    
    json_pattern = r'"topics"\s*:\s*(?P<topics>\[\s*"(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")*\s*\])'
    
    def use(self, message, topics):
        for topic in topics:
            if topic not in config.topic_sources:
                logger.warning(f'unknown topic: {topic}')
                continue
            logger.info(f'appending doc for topic: {topic}')
            doc = Document(
                page_content=Path(config.topic_sources[topic]).read_text())
            doc.metadata['important'] = True
            self._heymans.documentation.append(doc)
        return None, False
