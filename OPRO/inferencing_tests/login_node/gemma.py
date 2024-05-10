from transformers import pipeline
import torch, os, dotenv

dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
# Loading Model and creating pipeline
model = "../llama3-8b"
# model = "gemma_model_saved"
# model = "google/gemma-2b-it"

# Print model dir walk if model is a directory
if os.path.isdir(model):
    print(f"Model: {model}")
    print("Model Dir Walk:")
    for root, dirs, files in os.walk(model):
        for file in files:
            print(os.path.join(root, file))
else:
    print(f"Model: {model}")
    print("Model is not a directory.")

pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    # device="cuda",
    token=HUGGINGFACE_TOKEN,
    device_map="auto"
)

def generate_response(prompt):
    """
    Generate response for the given prompt.
    """
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        add_special_tokens=True,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    return outputs[0]["generated_text"][len(prompt):]

print(f"Q. What is the capital of USA?; A. {generate_response('What is the capital of USA?')}")
print(f"Q. What is 2+2?; A. {generate_response('What is 2+2?')}")
print(f"Q. Proton vs. Electron?; A. {generate_response('Proton vs. Electron?')}")
print(f"Q. Amalgum alloy used by dentists. What is the chemical composition?; A. {generate_response('Amalgum alloy used by dentists. What is the chemical composition?')}")

