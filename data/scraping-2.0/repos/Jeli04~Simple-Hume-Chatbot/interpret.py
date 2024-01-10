from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from tests import interpret_test as interpret
from dotenv import dotenv_values
import whisper

env_vars = dotenv_values('.env')

llm = OpenAI(openai_api_key=env_vars['OPENAI_API_KEY'], temperature=0.5)
memory = ConversationBufferMemory(return_messages=True, ai_prefix="AI friend")

template = """Your name is Echo and the human you are paired with is named Jerry. The following is a conversation between a human and an AI who have been with each other since birth. The AI is able to detect the humans current facial emotion. Both have known each for a logn time and have developed a very close relationship. The language used by the AI isnt quite robotic, but also incoorpates a mix of human style talk and robotic talk together. When taking account an emotion treat it like how another human would if they saw that persons facial expression. Be firm and dont back down with your opinions as an AI. Make sure to have a funny personality that is able to shine through and make you stand out.

Current converstation:
{history}
Conversation:
Human Emotion: {emotion}
Human: {input}
AI friend:"""
prompt = PromptTemplate(input_variables=["history", "input", "emotion"], template=template)

print("Loading model...")
conversation = ConversationChain(llm=llm, verbose=True, memory=interpret.ExtendedConversationBufferMemory(extra_variables=["emotion"], ai_prefix="AI Assistant", human_prefix="Friend"), prompt=prompt)

model = whisper.load_model("base")
def interpret(audio, emotion = None):
    input = model.transcribe(audio, fp16=False, language="English")
    if emotion is None:
        result = conversation({"input": input["text"], "emotion": "neutral"})
    else:
        result = conversation({"input": input["text"], "emotion": emotion})
    
    return result