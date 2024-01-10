"""Contains slack loaders used for loading data from slack into the vectorstore."""
import logging
import sys

from langchain.schema.vectorstore import VectorStore

from pulse import config
from pulse.services.slack import SlackEventType

logging.basicConfig(level=config.log_level, stream=sys.stdout)
logger = logging.getLogger(__name__)


async def event_loader(event: SlackEventType, vectorstore: VectorStore):
    """
    Load slack events into the vectorstore.
    """

    logger.debug("got event: %s", event)

    if event.type != "message":
        logger.info("Ignoring event of type %s", event.type)
        return

    text = event.text
    meta = {k: v for k, v in event.__dict__.items() if k != "text"}
    meta["source"] = "slack"

    logger.debug("Adding message to vectorstore: %s, {%s}", text, meta)

    id = await vectorstore.aadd_texts(texts=[text], metadata=meta)
    logging.debug("Added message to vectorstore: %s", id)
    return id
