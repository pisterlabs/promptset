import os, json, dotenv
from requests import ReadTimeout
from sentence_transformers import SentenceTransformer, util
import ollama

dotenv.load_dotenv()
modelfile='''
FROM {model}
PARAMETER temperature {temperature}
'''

def filecache(fn):
    filename = f".cache/{fn.__name__}.json"
    if not os.path.exists(".cache"):
        os.makedirs(".cache")

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump({}, f)

    with open(filename, "r") as f:
        cache = json.load(f)

    def wrapped(*args, **kwargs):
        # if optimize is set to True, the function will be called without caching
        if kwargs.get("is_indeterministic", False):
            return fn(*args, **kwargs)

        key = str(list(map(str, args))) + json.dumps(kwargs, sort_keys=True)
        if key not in cache:
            res = fn(*args, **kwargs)
            cache[key] = res
            with open(filename, "w") as f:
                json.dump(cache, f)

        return cache[key]

    return wrapped


def ollama_generate(lm, prompt, num_predict=-1):
    temp = lm.timeout
    while True:
        try:
            res = (
                lm.invoke(prompt)
                if num_predict < 0
                else lm.invoke(prompt, num_predict=num_predict)
            )
            break
        except ReadTimeout:
            if lm.timeout > 120:
                print(f"Inference lasted for {lm.timeout} seconds. Stopping now.")
                break
            lm.timeout *= 2
            print(
                f"### ReadTimeout. Trying again with Timeout: {lm.timeout} seconds ###"
            )
        except Exception as e:
            print(f"### {e} ###")
            break
    lm.timeout = temp
    return res


class OPRO:
    def __str__(self):
        return "OPRO"

    def __init__(self, init):
        # Load transformer model
        self.transformer_model = SentenceTransformer(
            "all-mpnet-base-v2"
        )  # Load transformer model

        # Load LMs
        if "gemini" in init:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

            self.gemini_model = genai.GenerativeModel("gemini-pro")

        if "gemma" in init:
            # from langchain_community.llms import Ollama

            # self.gemma_model = Ollama(
            #     model="gemma:2b", temperature=0, num_gpu=40, timeout=30
            # )
            ollama.create(model='gemma:2b_TEMP0', modelfile=modelfile.format("gemma:2b", 0))
            ollama.create(model='gemma:2b_TEMP1', modelfile=modelfile.format("gemma:2b", 1))

        
        if "llama2" in init:
            from langchain_community.llms import Ollama

            self.llama2_model = Ollama(
                model="llama2:7b", temperature=1, num_gpu=40, timeout=30
            )
        
        if "llama3" in init:
            ollama.create(model='llama3_TEMP0', modelfile=modelfile.format("llama3", 0))
            ollama.create(model='llama3_TEMP1', modelfile=modelfile.format("llama3", 1))

        if "anthropic" in init:
            import anthropic

            self.anthropic_client = anthropic.Anthropic()

    @filecache
    def generate(self, prompt, model="llama3", is_indeterministic=False, max_token=-1):
        temperature = float(is_indeterministic)  # 0 is default

        # gemini form
        if model == "gemini":
            import google.generativeai as genai
            generation_config = genai.GenerationConfig(temperature=temperature, max_output_tokens=2048)
            # WARNING: for some reason, gemini doesn't like it if output length exceeds token limit
            if max_token > 0:
                generation_config.max_output_tokens = max_token
            return (
                self.gemini_model.generate_content(
                    prompt, generation_config=generation_config
                )
                .candidates[0]
                .content.parts[0]
                .text
            )
        elif model == "gemma":
            # self.gemma_model.temperature = temperature
            # return ollama_generate(self.gemma_model, prompt, num_predict=max_token)
            return ollama.generate(model=f'gemma:2b_TEMP{int(temperature)}', prompt=prompt)["response"]
        elif model == "llama2":
            self.llama2_model.temperature = temperature
            return ollama_generate(self.llama2_model, prompt, num_predict=max_token)
        elif model == "llama3":
            return ollama.generate(model=f'llama3_TEMP{int(temperature)}', prompt=prompt)["response"]
        elif model == "anthropic":
            import anthropic

            try:
                return (
                    self.anthropic_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=max_token if max_token > 0 else 200,  # was initially 200
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    .content[0]
                    .text
                )
            except anthropic.InternalServerError:
                return self.generate(prompt, model="anthropic")

        raise ValueError("Invalid synth value")

    def similarity(self, text1, text2):
        embeddings = self.transformer_model.encode([text1, text2])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
