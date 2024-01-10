from dotenv import load_dotenv
load_dotenv()

from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory
import json

class ProcessTranscript:
    def __init__(self) -> None:
        template = """
        You should give a good summary of the transcript. 
        There will be multiple transcripts of the same video sent to you.
        
        {history}

        Human: {text}

        AI: 
        """

        prompt = PromptTemplate(
            input_variables=['text', 'history'], 
            template=template
        )
        llm = OpenAI(temperature=0)
        self.chatgpt_chain = LLMChain(
            llm=llm, 
            prompt=prompt, 
            verbose=True, 
            memory=ConversationSummaryMemory(llm=llm),
        )

    def summary(self, text):
        output = self.chatgpt_chain.predict(
            text=text
        )
        return output

if __name__ == "__main__":
    model = ProcessTranscript()
    output = None
    for i in range(10):
        with open(f"data/transcript_{i}.json", "r") as file:
            output = model.summary(
                json.loads(file.read())["text"]
            )
            with open(f"data/output_{i}.txt", "w") as file:
                file.write(output)

