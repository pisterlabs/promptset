from elevenlabs import generate, stream
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage, 
    SystemMessage)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI
import configs
import logging

llm = OpenAI(openai_api_key=configs.openAiKey)

def synthesize_voice(text: str):
    """Convert transcription to audio"""
    try:
        prediction = _synthesize_response(text=text)
        audio = generate(
            text=prediction,
            voice="Bella",
            model="eleven_multilingual_v2",
            streaming=True,
        )
        return stream(audio)
    except Exception as e:
        logging.error(f"Error: openai: {e}")
        raise e
    

def _synthesize_response(text: str):
    try:
        chat = ChatOpenAI(openai_api_key=configs.openAiKey,streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
        # 
        answer= chat([HumanMessage(content=text), SystemMessage(content="")])
        print("Synthesizing response...")
        if answer[0].content is None: 
            default_response = "I'm sorry, but I couldn't understand your query. Please rephrase it, and I'll do my best to help you."
            answer = [SystemMessage(content=default_response)]
            yield answer[0].content
        # Return the message from the system
        return answer[0].content
    except Exception as e:
        logging.error(f"Error: elevenlabs: {e}")
        raise e
    

