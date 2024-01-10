import gradio as gr
from langchain import LangChain  # Check the correct import based on langchain documentation

# Assuming you have the necessary configuration for langchain
langchain_api_key = "YOUR_LANGCHAIN_API_KEY"
langchain_model_name = "mistral"  # Replace with the actual model name

# Initialize langchain
langchain = LangChain(api_key=langchain_api_key, model=langchain_model_name)

history_conversation = []

def response_generate(prompt):
    history_conversation.append(prompt)

    full_prompt = "\n".join(history_conversation)

    # Assuming langchain provides a generate_text method or similar
    actual_response = langchain.generate_text(full_prompt)

    history_conversation.append(actual_response)
    return actual_response

def main():
    iface = gr.Interface(
        fn=response_generate,
        inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
        outputs="text"
    )

    iface.launch()

if __name__ == "__main__":
    main()
