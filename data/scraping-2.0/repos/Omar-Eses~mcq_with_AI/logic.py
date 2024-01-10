# essential imports in program
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# take user input -> add it to prompt -> chain -> pass to model (chatgpt) -> return model -> show it on streamlit

# create quiz prompt
def generate_questions(quiz_topic="programming in python", num_of_questions=1):
    llm = OpenAI()
    # prompt to involve the user in the model
    prompt_template = PromptTemplate(
        input_variables=["quiz_topic", "num_of_questions"],
        template="""
        you are an expert making multiple choice questions in the following topic: {quiz_topic}.
        create a multiple choice quiz that consists of {num_of_questions} questions,
        each question should have four options one of them is correct and you should send it to me.
        the correct answer should be marked with ^
        format it in the known multiple choice quiz template; example:
        Q1. what is the most famous programming language in the world?
            a. javascript
            b. Java
            c. C++
            d. Python
        Q2. what is the most famous language of the following?
            a. arabic
            b. english
            c. german
            d. russian
            - Answers:
            <Answer1>: a
            <Answer2>: b
        """
    )
    # Chaining the prompt so that it can be used to generate questions
    questions_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="question_format")

    response = questions_chain({'quiz_topic': quiz_topic, 'num_of_questions': num_of_questions})

    # # Parse the response to get questions and answers
    # parser = StrOutputParser()
    # parsed_response = parser(response)
    #
    # # Extract questions and answers
    # questions = [item['content'] for item in parsed_response['output']['content']]
    # answers = [item['answer'] for item in parsed_response['output']['content']]

    return response


if __name__ == "__main__":
    print(generate_questions("muscle hypertrophy", 2))
