import os
import datbase
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
response_cache = {}

def user(user_question,subject):
    summary=datbase.importing(subject)
    print(summary)
    def scheckpromptmaker(question):#for double checking g=for summary
        instructions = """You are a chatbot that returns 0 or 1.1 means yes and 0 means no.
        you should say 1 if the question is related to the summary or if user asks something related to what teacher taught that day and 0 if not.
    example of 1: "what did teacher teach today?"
    example of 1: "tell me about teachers class in short"
    """

        data = str(question) 
        question = '''Is the following question inquiring about the teacher's class or requesting a summary of the class content? Please return '1' if yes, and '0' if no.'''  
        
        sprompt = instructions + data + question

        return scheckaskgpt(sprompt)

    def scheckaskgpt(sprompt):
        
        chat_model = ChatOpenAI(temperature=0.1, model='gpt-3.5-turbo', openai_api_key=os.environ.get("OPENAI_API_KEY"), max_tokens=250, stop=["\n"])
        
        if sprompt in response_cache:
            return response_cache[sprompt]

        output = chat_model([HumanMessage(content=sprompt)])
        response2 = output.content
        response_cache[sprompt] = response2
        # print(response2)
        return response2


    def promptmaker(summary, question):
        instructions = """You are a chatbot who helps students learn.
        The summary of the topic that teacher taught that day is '{summary}'.
        You should help them answer all their questions with data mentioned in the '{summary}'.
        And  only answer the questions related to topics mentioned in  '{summary}'.
        If students ask anything outside the topic, you should say 'Sorry, I only know what the teacher taught.Do you have any other doubt related to the topic what teacher taught'.
        Don't ever mislead students with wrong answers or anything other than the {summary}.
        ask a creative question at the end of the answer to make students think more about the topic.
        Make sure you give an answer to the question in a way that students can understand easily.
        Don't give too long answers and go too deep into the answer.
        Don't forget to give a 'do you know' type question that invokes curiosity in students.
        if question is a casual talk like 'how are you' or 'what is your name' you should say 'I am fine' or 'my name is Studiesy-Your learnBuddy,You can ask me anything related to what teacher taught.' respectively.
        if question id hi or hello you should say 'Hello Buddy,'I am Studiesy-Your learnBuddy,You can ask me anything related to what teacher taught.''.
        Expected output example:
                        Photosynthesis is a process by which plants make their own food using sunlight, carbon dioxide and water. This process is an endothermic reaction and takes place in the chloroplasts of green plants. In this process, light energy is absorbed by chlorophyll and converted into chemical energy. This chemical energy is used to make glucose from carbon dioxide and water. Oxygen is also released as a by-product.
                        By the way, do you know about when it will take place?
            this is  only an example don't take this as a data for your answer.
    """

        data = str(summary)  # Convert summary to a string
        question = str(question)  # Convert question to a string
        
        # Check if the student asked about the topic taught that day
        if scheckpromptmaker(question) == "1":
            # print("The question is related to the summary")
            instruction=str(instructions)+" Always start with 'today teacher taught about'"
            prompt = instruction + data + "what is it about in short ?"

        else:
            # Use the default prompt for other questions
            prompt = instructions + data + question

        return askgpt(prompt)

    def askgpt(prompt):
        
        chat_model = ChatOpenAI(temperature=0.1, model='gpt-3.5-turbo', openai_api_key=os.environ.get("OPENAI_API_KEY"), max_tokens=250, stop=["\n"])
        
        if prompt in response_cache:
            return response_cache[prompt]

        output = chat_model([HumanMessage(content=prompt)])
        response = output.content
        response_cache[prompt] = response
        return response

    response = promptmaker(summary, user_question)

    return response



