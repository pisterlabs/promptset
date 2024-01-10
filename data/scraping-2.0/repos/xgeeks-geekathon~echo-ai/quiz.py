from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def createQuestions(transcript, num_open_question, multiple_option_question):
    print("Generating questions...")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    promptTemplate = PromptTemplate(
        template = """
        Given the following transcript:

        transcript: {transcript}

        Create {num_open_question} simples questions and answers and {multiple_option_question} multiple choice questions.

        The output should be in the following format "\n":
        
        Question (question number): (question)
        Answer: (answer)
        """,
        input_variables=["transcript","num_open_question","multiple_option_question"]
    )

    system_message_prompt = SystemMessagePromptTemplate.from_template("You are a quiz master that provide good simple questions based on the transcript.")
    human_message_prompt = HumanMessagePromptTemplate(prompt=promptTemplate)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chat_messages = chat_prompt.format_prompt(transcript=transcript, num_open_question=num_open_question, multiple_option_question=multiple_option_question).to_messages()

    result = llm(chat_messages)

    return result.content