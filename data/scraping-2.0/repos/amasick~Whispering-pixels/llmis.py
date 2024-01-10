from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from whisperiss import listening,speak
import warnings
from yolois import eyes
from dotenv import load_dotenv
from clip import generate_captions

load_dotenv()
warnings.filterwarnings("ignore",category=UserWarning,module='whisper.transcribe',lineno=114)
warnings.filterwarnings("ignore",message = "FP16 is not supported on CPU; using FP32 instead")
listis="Listening"
while True:
    lis="Do you wanna see or listen? "
    print(lis)
    speak(lis) 
    propis = listening()
    print(propis)
    if "listen" in propis.lower().strip():
        print("Listening...")
        speak(listis)
        prop = listening() 
        llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")
        if "use google" in prop.lower().strip():
            toolsis = load_tools(["google-serper"])
            # toolsis = load_tools(["google-serper","chatgpt"])
            agent = initialize_agent(toolsis, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
            print("Transcribed Audio:", prop)
            response = agent.run(prop)
            print("\033[34m"+"Generated Response: "+ response+"\033[0m")
            speak(response)
            
        else:
            # llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")
            summ = PromptTemplate(input_variables=[], template=prop)
            print(prop)
            chain = LLMChain(llm=llm, prompt=summ)
            response = chain.run(name=prop)
            print("\033[34m"+"Generated Response: "+ response+"\033[0m")
            speak(response)
            
    elif "exit" in propis.lower().strip():
        break
    else:
        speak("Seeing")
        eyes()
        txt1=generate_captions(r'captured_image.jpg')
        print("Listening...")
        speak(listis)
        prop = listening() 
        promptis= f" Given one description of the scene as {txt1}  . Treat all the answers for the same scene but with different details. Use all the details and answer the question using the above information as detailed and as accurately according to the information provided as possible. Donot make any assumptions from your side. {prop}"
        llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")
        print("Transcribed Audio:", prop)
        if "use google" in promptis.lower().strip():
            toolsis = load_tools(["google-serper"])
            agent = initialize_agent(toolsis, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
            response = agent.run(promptis)
            print("\033[34m"+"Generated Response: "+ response+"\033[0m")
            speak(response)
        else:  
            summ = PromptTemplate(input_variables=[], template=promptis)
            chain = LLMChain(llm=llm, prompt=summ)
            response = chain.run(name=prop)
            print("\033[34m"+"Generated Response: "+ response+"\033[0m")
            speak(response)

