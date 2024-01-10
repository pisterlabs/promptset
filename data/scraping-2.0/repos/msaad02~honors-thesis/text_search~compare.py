from text_retriever_class import TextRetriever
from typesense_retrieval import TypesenseRetrieval
from openai import OpenAI

class Compare():
    def __init__(
            self,             
            system_prompt=None,
            prompt=None,
            main_categorization_model_dir="./model",
            subcategorization_model_dir="./subcat_models/",
            embeddings_file="./data/embeddings.pickle",
        ):
        self.typesense_retrieval = TypesenseRetrieval(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,

        )

        self.old_retrieval = TextRetriever(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,
            embeddings_file=embeddings_file
        )

        self.client = OpenAI()

        self.system = """You are a helpful chatbot for SUNY Brockport who answers questions using the context given. Be enthusiastic, straightforward, and brief in your responses. Do not answer questions unrelated to SUNY Brockport. If the answer is not clear from the context, say "I'm sorry, I don't know".""" if system_prompt is None else system_prompt

        self.prompt = lambda context, question: f"Context: {context}\n\nQuestion: {question}" if prompt is None else prompt

    def compare(self, query):
        typesense_results = self.typesense_retrieval.ask(query)
        old_results = self.old_retrieval.retrieve(query)

        return {'typesense': typesense_results, 'old': old_results}
    
    def ask_openai(self, query, context, model="gpt-3.5-turbo-1106"):

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.prompt(context, query)},
            ],
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.2
        )

        return response.model_dump()['choices'][0]['message']['content']
    
    def rag(self, query, return_context: bool=False):
        "Ask a model a question based on the context"

        context = self.compare(query)

        # Typesense
        typesense_response = self.ask_openai(query, context['typesense'])

        # Old
        old_response = self.ask_openai(query, context['old'])

        if return_context:
            return {
                'typesense': typesense_response,
                'old': old_response,
                'typesense_context': context['typesense'],
                'old_context': context['old']
            }
        else:
            return {
                'typesense': typesense_response,
                'old': old_response
            }
