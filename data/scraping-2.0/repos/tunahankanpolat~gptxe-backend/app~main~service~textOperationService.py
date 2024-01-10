from app.main.model.query import Query
from app.main.dataAccess.queryDao import QueryDao
from app.main.utils.prompt import Prompt
from openai import ChatCompletion
class TextOperationService:
    def __init__(self, queryDao : QueryDao, prompt: Prompt, chatCompletion: ChatCompletion, maxToken: int) -> None:
        self.queryDao = queryDao
        self.prompt = prompt
        self.chatCompletion = chatCompletion
        self.maxToken = maxToken

    def getSummarizeContent(self, content):
        query = self.queryDao.getByRequest("summarize_content",content)
        tokenCount = self.prompt.getTokenCount(content)
        if not content:
            return {"error": "Content is empty."}, 400
        elif(query):
            return {"result": query, "message": "Request exists in cache"},200
        elif(tokenCount <= self.maxToken):
            try:
                response = self.chatCompletion.create(**self.prompt.getSummarizeContentPrompt(content))
                result = response.choices[0].message.content
                newQuery = Query("summarize_content", content, result)
                self.queryDao.addQuery(newQuery)
                return {"result": result, "message": "Successfully sent a request to the OpenAI Api"}, 200
            except Exception as error:
                return {"error": str(error.json_body.get("error").get("code"))}, error.http_status
        else:
            return {"error": "Your request "+ str(tokenCount) +" tokens exceeds the max token count of " + str(self.maxToken) +  "."}, 400
    
    def getFixTypos(self, content):
        query = self.queryDao.getByRequest("fix_typos", content)
        tokenCount = self.prompt.getTokenCount(content)
        if not content:
            return {"error": "Content is empty."}, 400
        elif(query):
            return {"result": query, "message": "Request exists in cache"},200
        elif(tokenCount <= self.maxToken):
            try:
                response = self.chatCompletion.create(**self.prompt.getFixTyposPrompt(content))
                result = response.choices[0].message.content
                newQuery = Query("fix_typos", content, result)
                self.queryDao.addQuery(newQuery)
                return {"result": result, "message": "Successfully sent a request to the OpenAI Api"}, 200
            except Exception as error:
                return {"error": str(error.json_body.get("error").get("code"))}, error.http_status
        else:
            return {"error": "Your request "+ str(tokenCount) +" tokens exceeds the max token count of " + str(self.maxToken) +  "."}, 400
    
    
    def getExplainCode(self, content, languagePreference = "English"):
        query = self.queryDao.getByRequest("explain_code", content)
        tokenCount = self.prompt.getTokenCount(content)
        if not content:
            return {"error": "Content is empty."}, 400
        elif(query):
            return {"result": query, "message": "Request exists in cache"},200
        elif(tokenCount <= self.maxToken):
            try:
                response = self.chatCompletion.create(**self.prompt.getExplainCodePrompt(content, languagePreference))
                result = response.choices[0].message.content
                newQuery = Query("explain_code", content, result)
                self.queryDao.addQuery(newQuery)
                return {"result": result, "message": "Successfully sent a request to the OpenAI Api"}, 200
            except Exception as error:
                return {"error": str(error.json_body.get("error").get("code"))}, error.http_status
        else:
            return {"error": "Your request "+ str(tokenCount) +" tokens exceeds the max token count of " + str(self.maxToken) +  "."}, 400