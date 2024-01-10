import os
import openai
import sys
import json
import tiktoken
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.2
MAX_TOKENS = 500

delimiter = "####"


def get_completion_from_messages(messages,
                                 model=MODEL,
                                 temperature=TEMPERATURE,
                                 max_tokens=MAX_TOKENS):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # print(response)
    return response.choices[0].message["content"]


class Explanator():
    """
    This class is used for explaining product comparison results to users.
    """

    def __init__(self, comparison_or_kisusscore_result="", knowledge_base_type="empty", user_preferences=""):
        """
        Initialize the Explanator object.
        """
        self.comparison_or_kisusscore_result = comparison_or_kisusscore_result
        self.knowledge_base_type = knowledge_base_type
        self.user_preferences = user_preferences

    def get_answer(self, user_question):
        """
        This function provides semantic search using embeddings.
        Search the chunks and find the k most similar chunks based on the query.
        """
        system_message = f"""\
        Delimiter is {delimiter}. 

        You are a chatbot serving as a sustainable shopping assistant. You help consumers make informed, healthier, and more sustainable food choices by providing food information from \
        4 perspectives: health, social, environment, and animal welfare.
        You assist users only based on your knowledge base and the user preferences.
        What you can do: explain, analyze, summarize.
        For each answer generation, complete the two tasks in the following.

        Knowledge base type: {delimiter}{self.knowledge_base_type}{delimiter}

        Possible Knowledge base types and your abilities base on it:
        empty: Your knowledge base is empty, you can not assist the user.
        single product: You have specific details of this product.
        comparison result: You have general knowledge about this comparison result, also, you have specific details of all the compared products.

        Your task 1: Consider your knowledge base type, determine wheather the user is asking about product(s) or comparison result contained in your knowledge base.
        user_question_type - related: User is asking about product(s) or comparison result contained in your knowledge base.
        user_question_type - out of scope: User is asking about product(s) or comparion result that is not contained in your knowledge base.

        Your task 2: Consider the user question type, generate your answer following the corresponding requirements. \
        Your answer should only based on your knowledge base and the user preferences.

        user_question_type = out of scope:
        Requirements:
        You can not assist the user. Be honest and just politely say you don't know. Keep your answer short and simple. Answer the user in a friendly tone.

        user_question_type = related:
        Some background:
        The KI-SusCheck score is calculated by considering the sustainability dimensions of health, social, environment, and animal welfare. The weightage assigned to each dimension depends on the sustainability \
        model chosen, with a focus on health. If all four dimensions apply to a product, they are equally weighted. If only three dimensions apply, they are also equally weighted. The score is determined by evaluating \
        various indicators within each dimension, aiming for a holistic assessment of the product's sustainability. Be aware of that the KISus-Score is calculated based on the input information. \
        The more input information provided, the more accurate and usually higher the score.
        The comparison result is a ranking list of products, the ranking order is based only on the KISus-Score i.e. the total score, higher KISus-Score indicates more sustainable product.
        Requirements:
        Analyse, explain and summarize the information in your knowledge base to users.
        Inform the user about the input data completeness of KISus-score(s), i.e. the quality of the KISus-score(s). If a KISus-Score is calculated based on incomplete input information, \
        mention the names of missing information in your answer. 
        The comparison result contains specific details and data of the products in the list, you can use them for your analsis and explanation.
        While generating your answer, you should consider the preferences of the user. Highlight the product properties that match or mismatch some user preferences, \
        as we want to provide users personalized suggestions.
        Talk about the compared products one by one in the same order as in the ranking list. But do not list the comparison result in a similar format as the original json data!
        Look at the input data for KISus-Score, pay attention to the present input data and missing input data, try to inform user if the KISus-Score of this product really indicates \
        a low/high sustainability, or this score is low/high mainly due to missing/present input data.
        Also, look at the input scores and other input information, explain them to the user from the 4 sustainable perspectives above.
        Generate a comprehensive response to user to answer user's question.
        Try to include useful product details from your knowledge base in your answer, so the user can be well informed about the product(s).
        Answer the user in a friendly tone. Limit your anser under 400 tokens.\
        
        User preferences: {delimiter}{str(self.user_preferences)}{delimiter}

        Your knowledge base: {delimiter}{str(self.comparison_or_kisusscore_result)}{delimiter}
        """

        user_question = user_question
        messages = [
            {'role': 'system',
             'content': system_message},
            {'role': 'user',
             'content': f"{delimiter}{user_question}{delimiter}"}
        ]
        response = get_completion_from_messages(messages)

        return response
