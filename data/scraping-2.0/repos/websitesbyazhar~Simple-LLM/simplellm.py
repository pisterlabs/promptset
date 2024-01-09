from flask import Flask, request, jsonify
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = Flask(__name__)

# Load the Llama model
n_gpu_layers = 1
n_batch = 512
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="D:\Programming Projects\ChatDoc\models\lamma2.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=True,
)

#Simple Server
@app.route('/generate', methods=['POST'])
def generate_text():
    if request.method == 'POST':
        data = request.get_json()
        prompt = data.get('prompt', '')
        generated_text = llm(prompt)
        return jsonify({"response": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)