from config.openai_setting import OpenAISetting
from config.pinecone_setting import PineconeSetting


class Setting:

    def __init__(self):
        OpenAISetting()
        PineconeSetting()
