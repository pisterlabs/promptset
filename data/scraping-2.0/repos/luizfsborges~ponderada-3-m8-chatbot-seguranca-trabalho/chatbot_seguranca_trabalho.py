from langchain.llms import Ollama
import gradio as gr

def run_ollama(text, ollama, chat_history):
    ollama_response = ollama(text)
    chat_history.append((text, ollama_response))
    return ollama_response, chat_history

def launch_app(base_url, model_name):
    ollama = Ollama(base_url=base_url, model=model_name)

    iface = gr.Interface(
        fn=run_ollama,
        inputs=gr.Textbox(),
        outputs=gr.Chatbox(),
        live=True,
        interpretation="ui",
        args=[ollama],
        examples=[["Quais EPIs são necessários para operar um torno mecânico?"]],
        title="Safety Expert Chatbot",
        description="Ask questions about safety standards in industrial environments.",
    )

    iface.launch()

if __name__ == "__main__":
    print("Loading Safety Expert Chatbot...")
    launch_app(base_url='http://localhost:11434', model_name="safety_expert")