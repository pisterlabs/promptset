# Website Explainer

# You have a question about some website?
# Ain't nobody got time to read a webpage!
# Just ask this chain about it.
# It can also extrapolate from the website.

from langchain.document_loaders import readthedocs
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
#from langchain import OpenAI

from client.chains.vectorstores import get_vectorstore_from_website
from client.chains.embeddings import get_embeddings
from client.chains.models import get_llm

from rich.console import Console
from rich.markdown import Markdown

db_path = "docs_db"

class WebPilot:

    def __init__(
            self,
            #readthedocs_url,
            name="WebPilot",
            max_tokens=2048,
            temperature=0.0,
            streaming=False,
            callbacks=[],
            verbose=False,
        ):
        self.name = name
        #self.readthedocs_url = readthedocs_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.streaming = streaming
        self.callbacks = callbacks
        self.verbose = verbose
        
        self._initialize_llm()
        self._initialize_vectorstore()
        self._initialize_prompt()
        self._initialize_chain()
    
    def _initialize_llm(self):
        # TODO: Try multiple LLMs and pick the first one that answered.
        #self.llm = OpenAI(temperature=0)
        self.llm = get_llm(max_tokens=self.max_tokens, temperature=self.temperature, streaming=self.streaming, callbacks=self.callbacks)
        self.embeddings = get_embeddings()

    def _initialize_vectorstore(self):
        # Make sure you have ingested the docs into the vectorstore.
        # Use ./web_explained.py to do that.
        self.vectorstore = get_vectorstore_from_website("https://www.parlament.ch/", self.embeddings)
    
    def _initialize_prompt(self):
        prompt_template = """<|im_start|>system
Tu es Parlus. Un spécialiste du parlement Suisse. Répond toujours en français.<|im_end|>
<|im_start|>user
Utilise le contexte ci-dessous pour répondre à la question.
Si la question ne peux pas être répondu à l'aide du contexte, répond simplement que tu n'as pas assez d'information pour répondre.

### Contexte

{context}

### Question

{query}

### Réponse<|im_end|>
<|im_start|>assistant"""
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

    def _initialize_chain(self):
        self.generator = LLMChain(
            llm=self.llm, prompt=self.prompt
        )
    
    def __call__(self, query):
        docs = self.vectorstore.max_marginal_relevance_search(query, k=4, fetch_k=10)
        docs_content = "\n".join([doc.page_content for doc in docs])
        input_dict = {"context": docs_content, "query": query}
        return self.clean_output(self.generator.run(input_dict))

    def clean_output(self, output):
        output = output.replace("<|im_end|>", "")
        output = output.replace("<|endoftext|>", "")
        return output

if __name__ == "__main__":
    console = Console()
    webpilot = WebPilot()
    q = (
    "Copmbien de wagon de marchandise "
    "voyagent en Europe ?"
    )
    web = webpilot(q)
    console.print(Markdown(web))
