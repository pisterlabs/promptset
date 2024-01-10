import speech_recognition as sr
import os
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_user_input(prompt, valid_inputs):
    while True:
        user_input = input(prompt).lower()
        if user_input in valid_inputs:
            return user_input
        else:
            print(f"Invalid input. Please enter one of the following: {', '.join(valid_inputs)}")

def get_voice_input():
    print("Press Enter and then speak your question:")
    input()

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        question = r.recognize_google(audio)
        print(f"You said: {question}")
        return question
    except sr.UnknownValueError:
        print("Sorry, I could not understand your voice.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

template = """Question: {question}

Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

TEMP = float(os.environ.get("TEMP", 0.6))
NCTX = os.environ.get("NCTX", 2500)

llm = LlamaCpp(
    model_path="/Users/oluwaseunolaoya/Documents/LLM_Models/ggml-gpt4all-l13b-snoozy.bin",
    callback_manager=callback_manager,
    verbose=True,
    temperature=TEMP,
    n_ctx=NCTX,
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

print("Welcome to the chatbot! You can ask questions by typing or speaking.")
print("Type 'quit', 'exit', or 'stop' to end the session.")

while True:
    input_method = get_user_input("Would you like to type or speak your question? (type/speak): ", ["type", "speak"])

    if input_method == "speak":
        question = get_voice_input()
        if not question:
            continue
    else:
        question = input("Please type your question: ")

    if question.lower() in ["quit", "exit", "stop"]:
        break

    output = llm_chain.run(question)

    sanitized_output = output.replace("'", '"').replace("\n", " ").strip()
    
    #os.system("say '" + sanitized_output + "'")
