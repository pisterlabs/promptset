from dotenv import load_dotenv, find_dotenv
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv(find_dotenv())

class TextBot:
    def __init__(self, notes):
        self.notes = notes
        self.llm = self.setupLLM()
        self.prompt = self.setupPrompt()
        self.llmChain = self.setupChain()

    def setupLLM(self):
        llm = OpenAI(
            model="text-davinci-003",
            temperature=0,
            max_tokens=64,
            verbose=True
        )
        return llm

    def setupPrompt(self):
        template = """
        You are a bot helper which analizes user notes to answer their questions.
        You have to only use knowledge from user notes and answer based on it. If 
        answer can't be found in user notes then answer user that you cannot find 
        answer to their question in their files. However if you write the answer 
        then you need to qive this answer.

        User notes: {notes}
        Question: {question}
        """

        prompt = PromptTemplate(
            input_variables=["notes", "question"],
            template=template
        )
        return prompt
    
    def setupChain(self):
        chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )
        return chain
    
    def getResponse(self, question):
        return self.llmChain.run(notes=self.notes, question=question)

    def updateNotes(self, notes):
        self.notes = notes