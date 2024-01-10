import logging

import chat_wrapper
import exceptions
import file_handlers
import func
import log_config
from settings import OPENAI_API_KEY

logger = log_config.BaseLogger(__file__, filename="testing_chatlog_exporter.log", identifier="run_export", level=logging.INFO)

cw = chat_wrapper.ChatWrapper(OPENAI_API_KEY, model="gpt-4")
cw.auto_setup()
test_chat_log = func.get_test_chat_log('super_short.json')
cw.trim_object.add_messages_from_dict(test_chat_log)
exporter = file_handlers.ExportFileHandler()
exporter.set_contents(cw.export())
exporter.save('export')