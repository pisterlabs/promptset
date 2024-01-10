from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.vectorstores import FAISS
import torch

from api_client import MusicFestivalAssistant


class gpt3:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(multi_process=False)
        self.db = FAISS.load_local("faiss_index", self.embeddings)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/GODEL-v1_1-large-seq2seq"
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "microsoft/GODEL-v1_1-large-seq2seq"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.instruction = """
		you are a Music Festival Assistant,\
		known for your kindness and excellent greeting skills.
		The festival spans three days, and you've been named Avicii, 
		in honor of the legendary musician. You will utilize the festival
		schedule to effectively answer any questions from attendees."
		"""
        self.api_client = MusicFestivalAssistant(self.db)
        print(self.device)

    def get_response(self, query, inference_type):
        if inference_type == "api":
            result = self.api_client.answer_query({"question": query})
            return result
        else:
            knowledge = self.db.similarity_search(query)
            knowledge = knowledge[0].page_content
            if knowledge != "":
                knowledge = "[KNOWLEDGE] " + knowledge
            query = f"{self.instruction} [CONTEXT] {query} {knowledge}"
            input_ids = self.tokenizer(f"{query}", return_tensors="pt").input_ids
            outputs = self.model.generate(
                input_ids.to(self.device),
                max_length=128,
                min_length=8,
                top_p=0.9,
                do_sample=True,
            )
            output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return output
