import os, json, dotenv
from requests import ReadTimeout

dotenv.load_dotenv()


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


def ollama_generate(lm, prompt):
    temp = lm.timeout
    while True:
        try:
            res = lm.invoke(prompt)
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
        if "gemini" in init:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

            self.gemini_model = genai.GenerativeModel(
                "gemini-pro",
                generation_config=genai.GenerationConfig(
                    temperature=0, max_output_tokens=2048
                ),
            )

        if "gemma" in init:
            from langchain_community.llms import Ollama

            self.gemma_model = Ollama(
                model="gemma:2b", temperature=0, num_gpu=40, timeout=30
            )

        if "anthropic" in init:
            import anthropic

            self.anthropic_client = anthropic.Anthropic()

    @filecache
    def generate(self, prompt, model="gemini", is_indeterministic=False):
        temperature = float(is_indeterministic)

        # gemini form
        if model == "gemini":
            return (
                self.gemini_model.generate_content(
                    prompt, generation_config={"temperature": temperature}
                )
                .candidates[0]
                .content.parts[0]
                .text
            )
        elif model == "gemma":
            self.gemma_model.temperature = temperature
            return ollama_generate(self.gemma_model, prompt)
        elif model == "anthropic":
            import anthropic

            try:
                return (
                    self.anthropic_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=200,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    .content[0]
                    .text
                )
            except anthropic.InternalServerError:
                return self.generate(prompt, model="anthropic")

        raise ValueError("Invalid synth value")
