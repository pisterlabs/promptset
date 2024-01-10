import openai
from .ChatModelCompletor import get_completion_from_messages as get_completion_from_messages

class EmailSubjectGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def generate_subject(self, customer_comment, language="English"):
        system_message = f"""
            You are a customer service representative of the store.\
            You need to generate an email subject based on the customer's comment and in {language} language.\
            The user comment will be provided delimited with \
            #### characters.\
            Only generate an email subject, with nothing else.\
        """

        user_message_1 = f"""
            {customer_comment} \
        """

        messages =  [
            {'role':'system',
            'content': system_message},
            {'role':'user',
            'content': f"####{user_message_1}####"},
        ]

        return get_completion_from_messages(messages)

class CommentSummarizer:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def summarize_and_translate(self, customer_comment, target_language):
        system_message = f"""
            You are a customer comment summarizer of the store.\
            You need to summarize the following customer comment and translate it into {target_language} language.\
            The customer comment will be delimited with \
            #### characters.\
            Only generate a summary, with nothing else.\
        """

        user_message_1 = f"""
            {customer_comment} \
        """

        messages =  [
            {'role':'system',
            'content': system_message},
            {'role':'user',
            'content': f"####{user_message_1}####"},
        ]

        return get_completion_from_messages(messages)
    
class EmailGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def generate_email(self, customer_comment, comment_summary, sentiment, language, products):
        system_message = f"""
            You are a customer service representative of the store.\
            You need to generate an email based on the customer's comment, its summary, sentiment, and language.\
            The customer comment will be delimited with \
            #### characters.\
            The summary will be delimited with \
            ##### characters.\
            The sentiment will be delimited with \
            ###### characters.\
            Your email should be written in {language}.\
            The product info is {products}.\
            Only generate an email response, with nothing else.\
        """

        user_message_1 = f"""
            ####{customer_comment}#### \
            #####{comment_summary}##### \
            ######{sentiment}###### \
        """

        messages =  [
            {'role':'system',
            'content': system_message},
            {'role':'user',
            'content': f"{user_message_1}"},
        ]

        return get_completion_from_messages(messages)
    
    def chain_of_thought_reasoning_generate_email(self, customer_comment, comment_summary, sentiment, language, products):
        delimiter = "####"
        system_message = f"""
            Follow these steps to answer the customer queries.
            The customer query will be delimited with four hashtags,\
            i.e. {delimiter}. 

            # Step 1: deciding the type of inquiry
            Step 1:{delimiter} First decide whether the user is \
            asking a question about a specific product or products. \

            Product cateogry doesn't count. 

            # Step 2: identifying specific products
            Step 2:{delimiter} If the user is asking about \
            specific products, identify whether \
            the products are in the following list.
            All available products: {products}

            # Step 3: listing assumptions
            Step 3:{delimiter} If the message contains products \
            in the list above, list any assumptions that the \
            user is making in their \
            message e.g. that Laptop X is bigger than \
            Laptop Y, or that Laptop Z has a 2 year warranty.

            # Step 4: providing corrections
            Step 4:{delimiter}: If the user made any assumptions, \
            figure out whether the assumption is true based on your \
            product information. 

            # Step 5
            Step 5:{delimiter}: First, politely correct the \
            customer's incorrect assumptions if applicable. \
            Only mention or reference products in the list of \
            5 available products, as these are the only 5 \
            products that the store sells. \
            Answer the customer in a friendly tone.

            Use the following format:
            Step 1:{delimiter} <step 1 reasoning>
            Step 2:{delimiter} <step 2 reasoning>
            Step 3:{delimiter} <step 3 reasoning>
            Step 4:{delimiter} <step 4 reasoning>
            Response to user:{delimiter} <response to customer>

            Make sure to include {delimiter} to separate every step.
            Response in customers language {language}.
        """

        user_message_1 = f"""
            ####{customer_comment}#### \
        """

        messages =  [
            {'role':'system',
            'content': system_message},
            {'role':'user',
            'content': f"{user_message_1}"},
        ]

        return get_completion_from_messages(messages)

