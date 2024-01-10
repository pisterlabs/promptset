import uuid
import json
import openai
from model.data_store.mysql_driver import MySQLDriver
from model.objects.item import Item

class Service():

    def __init__(self, dbDriver=None):
        self.envData = self._load_env_data()
        openai.organization = self.envData["openai.organization"]
        openai.api_key = self.envData["openai.api_key"]
        if not dbDriver:
            dbDriver = MySQLDriver()
            dbDriver.init_connection("app_template")
        self.dbDriver = dbDriver

    def upsert_item(self, itemDict):
        item = Item()
        item.from_dictionary(itemDict)
        if not item.id:
            item.id = str(uuid.uuid4())
        executionResponse = None
        try:
            item, response = self.dbDriver.upsert(item)
            executionResponse = self.dbDriver.commit()
        except Exception as e:
            executionResponse = self.dbDriver.rollback()
            #print(e)
            #print(executionResponse)
        return item, executionResponse

    def filter_items(self, filters):
        item = Item()
        items, response = self.dbDriver.filter_multiple(item, filters)
        return items, response

    def connect_to_openai(self, messages):
        print(messages)
        openAIMessages = []
        for message in messages:
            openAIMessages.append({
                "role": "user",
                "content": message
            })
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=openAIMessages,
            top_p=self.envData["openai.top_p"]
        )
        print(completion)
        return completion

    def _load_env_data(self):
        fileName = "env_data.json"
        file = open(fileName)
        envDataText = file.read()
        envData = json.loads(envDataText)
        return envData
