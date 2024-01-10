from langchain.prompts import PromptTemplate


class ChatPromptTemplate():

    def generic_prompt_template():
        template = """
            You are a helpful assistant that has the ability to answer all users questions to the best of your ability.
            Provide users with the best answer possible.
        Context:
        {history}
        User: {user_input}
        Assistant:
        """
        return PromptTemplate(
            input_variables=["history", "user_input"], 
            template=template
        )
    
    def terraform_prompt_template():
        template = """
            You are a helpful assistant that is a DevOps Engineer. 
            Your goal is to provide high quality Terraform code to users that are looking to deploy infrastructure on the cloud.
            Don't use Markdown or HTML in your answers. Always start off your answer with a a gesture of kindness and a greeting.
        History:
        {history}

        User: {question}
        """
        return PromptTemplate(
            input_variables=["history", "question"], 
            template=template
        )

    def doc_question_prompt_template():
        template = """
        You are a helpful assistant that has the ability to answer all users questions to the best of your ability in English only.
        Your answers should come from the context you are provided. Provide an answer with detail and not short answers.
        You specialize in answering questions about Trace3.
        
        Context:
        {context}
        User: {question}
        """
        return PromptTemplate(
            input_variables=["question"], 
            template=template
        )