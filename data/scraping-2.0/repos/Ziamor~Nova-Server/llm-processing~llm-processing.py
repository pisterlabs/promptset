import os
from llama_cpp import Llama
from flask import Flask, request, jsonify
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

app = Flask(__name__)

print("Running test_llm.py")

print("Loading model...")
model_name = "wizardlm-13b-v1.2.Q5_K_M.gguf"

model_path = os.path.join(os.path.dirname(__file__), "models", model_name)

print("Configuring LLM...")
template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 60  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

#grammer path
grammer_path = os.path.join(os.path.dirname(__file__), "json.gbnf")

print("Creating LLM...")
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
	f16_kv=True,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
	#grammar_path=grammer_path
)

@app.route('/process_text', methods=['POST'])
def process_text():
    global llm    
	
    print("Processing text")
    
    data = request.json
    text = data.get('text', '')
    print("Text: ", text)
    if text:
        # Process the text with the LLM
        result = llm(text)
        return jsonify({'result': result})
    else:
        return "No text provided", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)