""" Class to connect to the local and external LLM services"""

from llama_cpp import Llama
import time
from dotenv import load_dotenv
import os
from openai import OpenAI
import openai



# Lade die Umgebungsvariablen aus der .env-Datei
load_dotenv()

# LLM settings for GPU
n_gpu_layers = 43  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# model_name = "spicyboros-13b-2.2.Q5_K_M.gguf"
# model_name = "Llama-2-13b-chat-german-GGUF.q5_K_M.bin"
model_name = "mistral-7b-openorca.Q5_K_M.gguf"

# "ON" or "OFF
LOAD_LLM = "ON"

# Prompt for the tagger
MISTRAL_TAGGER_PROMPT = '''Find three keywords matching the following text delimited by triple backquotes.
```{content}```
Write each of the three keywords with relevance scores in JSON Format with nothing else, no comments. Use "keyword" and "score" as field names.
THREE KEYWORDS IN JSON: '''

# Model for the tagger
TAGGER_MODEL_NAME = "mistral-7b-openorca.Q5_K_M.gguf"


class Jarvis:
    llm = None
    openai_client = None
    current_loaded_model = None
    
    
    # this function invalidates the current LLM model and loads a new one
    def switch_local_model(self, model_name):
        self.llm = None
        time_start = time.time()
        print("Loading model: %s" % model_name)
        # load the large language model file
        if (LOAD_LLM != "OFF"):
            self.llm = Llama(model_path="models/" + model_name,
                             n_ctx=4096,
                             n_gpu_layers=n_gpu_layers,
                             n_batch=n_batch,
                             verbose=True)
            time_to_load = time.time() - time_start
            print("RELOADED new model %s in % seconds" % (model_name, time_to_load))
            self.current_loaded_model = model_name
        else:
            self.llm = None
        return


    def tag_content(self, content):
        tag_string = None
        
        if (TAGGER_MODEL_NAME != self.current_loaded_model):
            self.switch_local_model(TAGGER_MODEL_NAME)

        
        prompt = MISTRAL_TAGGER_PROMPT.replace("{content}", content)
        # print ("** Tagger Start: ", repr(prompt))                
        time_start = time.time()
        output = self.llm(prompt,
                              max_tokens=150,
                              # stop=["</s>", ],
                              echo=False,
                              temperature=0.2,
                              top_p=0.5,
                              top_k=3,
                              )
        # print("** Tagger ready: ", repr(output))
        time_to_load = time.time() - time_start
        ptime = "%.1f" % time_to_load
        print("** TIME %s seconds" % ptime)
        tag_string = output["choices"][0]["text"]
                
        return tag_string


    

    def ask(self, model=None, prompt=None, context=None, question=None):
        if model is None:
            print("ERROR:: Jarvis.ask() No model specified.")
            raise Exception("ERROR:: Jarvis.ask() No model specified.")

        if 'model_type' not in model:
            print("ERROR:: Jarvis.ask() No model_type specified.")
            raise Exception("ERROR:: Jarvis.ask() No model_type specified.")

        if model['model_type'] == "local":
            if (model['model_filename'] != self.current_loaded_model):
                self.switch_local_model(model['model_filename'])
            
            return self.ask_local_model(prompt)
        elif model['model_type'] == "remote":
            return self.ask_remote_model(model=model, prompt=prompt, context=context, question=question)


    def ask_davinci_model(self, model=None, prompt=None, context=None, question=None):

        time_start = time.time()
        print("ask_davinci_model() model: %s" % (model['model_filename']))

        prompt.format(context=context, question=question)

        print("ask_davinci_model() prompt: %s" % prompt)

        completion = self.openai_client.completions.create(model="davinci", prompt=prompt, max_tokens=100, )

        answer = completion.choices[0].text
        time_query = time.time() - time_start

        print("ask_openai_model() answer: %s" % answer)
        print("Query executed in %s seconds" % time_query)

        return [answer, time_query]


    def ask_gpt_model(self, model=None, prompt=None, context=None, question=None):

        time_start = time.time()
        print("ask_gpt_model() model: %s" % (model['model_filename']))

        messages = [
            {"role": "system",
             "content": prompt},
            {"role": "user",
             "content": "Berücksichtige folgende Informationen, um die Frage des Benutzers zu beantworten: %s" % context},
            {"role": "assistant",
             "content": "Danke, ich werde diese Informationen bei meiner Antwort berücksichtigen."},
            {"role": "user",
             "content": "Beantworte folgende Frage: %s" % question},
        ]
        if not context or context == "":
            messages = [
                {"role": "system",
                 "content": prompt},
                {"role": "user",
                 "content": "Beantworte folgende Frage: %s" % question},
            ]

        completion = self.openai_client.chat.completions.create(
            model=model['model_filename'],
            messages=messages,
        )

        answer = completion.choices[0].message.content
        time_query = time.time() - time_start

        print("ask_openai_model() answer: %s" % answer)
        print("Query executed in %s seconds" % time_query)

        return [answer, time_query]

    def ask_remote_model(self, model=None, prompt=None, context=None, question=None):
        # if model.model_type contains "*gpt*" ignore uppercase then use OpenAI API
        if "gpt" in model['model_filename'].lower():
            return self.ask_gpt_model(model=model, prompt=prompt, context=context, question=question)
        elif "davinci" in model['model_filename'].lower():
            return self.ask_davinci_model(model=model, prompt=prompt, context=context, question=question)
        else:
            print("ERROR:: Jarvis.ask_remote_model() No valid model specified. %s" % model)
            raise Exception("ERROR:: Jarvis.ask_remote_model() No valid model specified %s." % model)        

    def ask_local_model(self, prompt):
        time_start = time.time()

        if not prompt or prompt == "":
            return ["Kein Prompt übergeben.", None]

        # prompt = """
        # <s>[INST] <<SYS>>You are a helpful, honest assistant.
        #  Use the following pieces of information to answer the user's question: {context} <</SYS>>
        #  {question}[/INST] This is a answer </s>
        # """

        print("start jarvis.ask(): %s" % prompt)

        if LOAD_LLM != "OFF":
            output = self.llm(prompt,
                              max_tokens=256,
                              # stop=["Q:", "\n"],
                              echo=False,
                              temperature=0.2,
                              top_p=0.5,
                              top_k=3,
                              )
            # print("** Answer ready ask(): ", repr(output))

        else:
            # print('Warning: LOAD_LLM == "OFF"')
            time.sleep(2)
            # print('Ende jarvis.ask()')
            time_query = time.time() - time_start
            return ["Das LLama Modell ist deaktiviert.", time_query]

        time_query = time.time() - time_start
        print("Query executed in %s seconds" % time_query)

        answer = output["choices"][0]["text"]

        return [answer, time_query]

    def __init__(self):
                                
        
        print("loading model: %s" % model_name)
        time_start = time.time()
        # load the large language model file
        if (LOAD_LLM != "OFF"):
            self.llm = Llama(model_path="models/" + model_name,
                             n_ctx=4096,
                             n_gpu_layers=n_gpu_layers,
                             n_batch=n_batch,
                             verbose=True)
            self.current_loaded_model = model_name
        else:
            self.llm = None

        time_to_load = time.time() - time_start
        print("loaded model %s in %s seconds" % (model_name, time_to_load))

        # load the OpenAI API
        openai_key = os.getenv('OPENAI_API_KEY')
        openai_org = os.getenv('OPENAI_API_ORG')
        openai.api_key = openai_key # Set API key
        self.openai_client = OpenAI(
            organization=openai_org,
            api_key=openai_key,
        )


