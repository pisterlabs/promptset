from transformers import pipeline
import torch, os, dotenv

dotenv.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

class LLM:
    def __init__(self, init):
        if "gemma-2b" in init:
            self.gemma_2b = "google/gemma-2b-it"
            self.gemma_2b_pipeline = pipeline(
                "text-generation",
                model=self.gemma_2b,
                model_kwargs={"torch_dtype": torch.bfloat16},
                token=HUGGINGFACE_TOKEN,
                device_map="auto"
            )

        if "llama3-8b" in init:
            self.llama3_8b = "meta-llama/Meta-Llama-3-8B-Instruct"
            self.llama3_8b_pipeline = pipeline(
                "text-generation",
                model=self.llama3_8b,
                model_kwargs={"torch_dtype": torch.bfloat16},
                token=HUGGINGFACE_TOKEN,
                device_map="auto"
            )

        if "llama3-70b" in init:
            self.llama3_70b = "meta-llama/Meta-Llama-3-70B-Instruct"
            self.llama3_70b_pipeline = pipeline(
                "text-generation",
                model=self.llama3_70b,
                model_kwargs={"torch_dtype": torch.bfloat16},
                token=HUGGINGFACE_TOKEN,
                device_map="auto"
            )

    def generate_response(self, prompt, model="gemma-2b", temperature=0.0):
        """
        Generate response for the given prompt.
        """
        if model == "gemma-2b":
            pipeline = self.gemma_2b_pipeline
        elif model == "llama3-8b":
            pipeline = self.llama3_8b_pipeline
        elif model == "llama3-70b":
            pipeline = self.llama3_70b_pipeline
        else:
            raise ValueError(f"Model {model} not found.")

        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            add_special_tokens=True,
            do_sample=bool(temperature),
            temperature=temperature,
        )
        return outputs[0]["generated_text"][len(prompt):]