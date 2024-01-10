import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer 
from optimum.onnxruntime import ORTModelForTokenClassification
from optimum.pipelines import pipeline
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
) 
import onnx
from dotenv import dotenv_values

env_variables = dotenv_values('.env')

OPENAI_API_KEY = env_variables['OPENAI_API_KEY']


class ContentCreatorChatbot:
    def __init__(self, openai_api_key: str):
        """
        Initializes the ContentCreatorChatbot instance.

        Args:
            openai_api_key (str): The OpenAI API key.
        """
        self.openai_api_key = openai_api_key
        self.nlp, self.stop_words, self.lemmatizer = self.load_resources()
        self.chat_prompt_template = self.create_chat_prompt_template()

    def load_resources(self) -> tuple:
        """
        Loads the required NLP resources.

        Returns:
            tuple: A tuple containing the loaded resources (nlp, stop_words, lemmatizer).
        """
        tokenizer = AutoTokenizer.from_pretrained("optimum/bert-base-NER")
        model = ORTModelForTokenClassification.from_pretrained("optimum/bert-base-NER")

        nlp = pipeline("token-classification", model=model, tokenizer=tokenizer)

        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        return nlp, stop_words, lemmatizer

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses the input text by performing text cleaning, stopwords removal, tokenization, and lemmatization.
        
        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        cleaned_text = text.lower().strip()
        tokens = word_tokenize(cleaned_text)
        tokens = [token for token in tokens if token not in self.stop_words]
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_sentence = ' '.join(lemmatized_tokens)
        ner_results = self.perform_ner(lemmatized_sentence)
        text = lemmatized_sentence + str(ner_results)
        return text

    def perform_ner(self, text: str) -> dict:
        """
        Performs Named Entity Recognition (NER) on the input text.

        Args:
            text (str): The text to perform NER on.

        Returns:
            dict: The NER results.
        """
        return self.nlp(text)

    def prepare_for_summerization(self, preprocessed_text: str, ner_results: dict) -> str:
        """
        Prepares the input for the chatbot.

        Args:
            preprocessed_text (str): The preprocessed text.
            ner_results (dict): The NER results.

        Returns:
            str: The prepared input for the chatbot.
        """
        return preprocessed_text + str(ner_results)

    def create_chat_prompt_template(self) -> ChatPromptTemplate:
        """
        Creates the chat prompt template.

        Returns:
            ChatPromptTemplate: The chat prompt template.
        """
        human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="""
                From the given description of a content creator generate a Concise and appealing summary for potential investors based on the description and ner results.
                Never produce lists, just a concise and appealing summary in one small paragraph of 200 words.
                {human_input}:
                """,
                input_variables=["human_input"],
            )
        )
        return ChatPromptTemplate.from_messages([human_message_prompt])

    def init_chatbot(self) -> str:
        """
        Initializes the chatbot and returns the initial message.

        Returns:
            str: The initial message from the chatbot.
        """
        return "Chatbot: Hello, This is Listian from Listed, I am here to help you. To better understand you as a content creator, Please provide your name, location, and educational background. and Could you please share more about the type of content you create, the platforms you use to share it, the size of your audience, the theme or focus of your content, any notable achievements you've had so far, your future plans, and what makes your approach unique? I'm eager to learn more about your work and what sets you apart as a content creator..."

    def run_chatbot(self) -> None:
        """
        Runs the chatbot interaction.
        """
        print("")
        print(self.init_chatbot())
        print("")
        chat = ChatOpenAI(temperature=0, openai_api_key=self.openai_api_key)
        chain = LLMChain(llm=chat, prompt=self.chat_prompt_template)
 
        human_input = input("User: ")
        preprocessed_text = self.preprocess_text(human_input)
        ner_results = self.perform_ner(preprocessed_text)
        input_text = self.prepare_for_summerization(preprocessed_text, ner_results)
        print("")
        print("Chatbot:",chain.run(input_text))
        print("")

if __name__ == "__main__":       
    chatbot = ContentCreatorChatbot(openai_api_key=OPENAI_API_KEY)
    chatbot.run_chatbot()