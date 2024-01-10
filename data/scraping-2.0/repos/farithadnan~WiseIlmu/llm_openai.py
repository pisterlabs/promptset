import pprint
import tiktoken
from halo import Halo
from omegaconf import DictConfig
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain


class LLMOpenAI:
    def __init__(self, cfg: DictConfig, temperature=None, max_tokens=None):
        if (temperature is None) or (max_tokens is None):
            self.temperature = cfg.openAI.temperature
            self.max_tokens = cfg.openAI.max_tokens
        else:
            self.temperature = temperature
            self.max_tokens = max_tokens

        self.api_key = cfg.openAI.api_key
        self.model = cfg.openAI.model
        self.chat_cost_per_1000_tokens = cfg.openAI.chat_cost_per_1000_tokens

    def get_llm(self):
        '''
        Method to get the LLM model.
        
        Returns:
            The LLM model.
        '''
        try:
            llm = ChatOpenAI(openai_api_key=self.api_key, model_name=self.model, 
                   temperature=self.temperature, max_tokens=self.max_tokens)
        except (AttributeError, KeyError) as e:
            raise ValueError(f"Invalid or missing configuration: {e}")
        
        return llm
    
    def get_qa_chain(self):
        '''
        Method to get the Q & A chain.

        Returns:
            The Q & A chain.
        '''
        spinner = Halo(text='Configuring Q & A Chain...', spinner='dots')
        spinner.start()  

        llm = self.get_llm()

        spinner.stop()
        return load_qa_chain(llm=llm)
    
    def generate_response(self, vector_db: Chroma, qa_chain: BaseCombineDocumentsChain, messages):
        '''
        Method to generate a response from the chatbot.

        Args:
            vector_db: The vector database.
            qa_chain: The Q & A chain.
            messages: The messages sent by the user.

        Returns:
            The chatbot's response.
        '''
        yellow = "\033[0;33m"
        print("\n")
        # Create a loading spinner
        spinner = Halo(text='Loading Response...\n', spinner='dots')
        spinner.start()

        # Fetch latest user Input
        latest_input = next((message for message in reversed(messages) if message.get("role") == "user"), None)

        # Get matching documents based on input text
        matching_docs = vector_db.similarity_search(latest_input["content"])
        answer = qa_chain.run(input_documents=matching_docs, question=messages)

        # Stop the spinner once the response is received
        spinner.stop()

        # Testing - Pretty-print the user message sent to the AI
        pp = pprint.PrettyPrinter(indent=4)
        print(f"{yellow}\n--------------------------------------------------")
        print(f"{yellow}💫 Request:")
        print(f"{yellow}--------------------------------------------------")

        pp.pprint(messages)


        # Testing - Pretty-print chat cost
        print(f"{yellow}--------------------------------------------------")
        print(f"{yellow}💰 Chat Cost:")
        print(f"{yellow}--------------------------------------------------")


        total_word_count, total_token_count, estimated_cost = self.estimate_cost(latest_input["content"], answer)

        print(f"{yellow}Total Word Count: {total_word_count}")
        print(f"{yellow}Total Token Count: {total_token_count}")
        print(f"{yellow}Estimated Cost: ${estimated_cost}")
        print(f"{yellow}--------------------------------------------------")


        return answer

    def estimate_cost(self, user_input: str, bot_response: str):
        '''
        Method to estimate the cost of a chat.
        
        Args:
            user_input (str): The user's input.
            bot_response (str): The chatbot's response.
        
        Returns:
            The estimated cost, total word count, and total token count.
        '''
        # Create a loading spinner
        spinner = Halo(text='Estimating Cost...\n', spinner='dots')
        spinner.start()

        # Get the total token count
        combined_text = user_input + " " + bot_response
        encoded = tiktoken.encoding_for_model(self.model)
        total_token_count = len(encoded.encode(combined_text))

        # Calculate estimated cost for chat
        estimated_cost = "{:.10f}".format(total_token_count * self.chat_cost_per_1000_tokens/1000)
        total_word_count = len(combined_text.split())

        spinner.stop()
        return total_word_count, total_token_count, estimated_cost
