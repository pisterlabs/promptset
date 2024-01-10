from langchain.llms import OpenAI
from langchain.chains import ConversationChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate


from langchain.chains.conversation.memory import ConversationBufferMemory

OPENAI_API_KEY = '#OPEN_API_KEY'
llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-4')
conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)


def generate_coding_question(conversation_buf, topics, difficulty):
    topics = ", ".join(topics)
    CodingQuesTemplate = "Can you create a coding question on the topic {topic} ?. Make the question {difficulty}. Only provide the question"
    TestCaseTemplate = "Can you write the test cases for the above question ?"
    CodingPrompt = PromptTemplate(
        input_variables=['topic', 'difficulty'], template=CodingQuesTemplate)
    question = conversation_buf(CodingPrompt.format(
        topic=topics, difficulty=difficulty))['response']
    test_cases = conversation_buf(TestCaseTemplate)['response']
    print(f"This is the question: {question}")
    print(f"This is the test_case: {test_cases}")
    return question, test_cases


def get_coding_answer(conversation_buf, question):
    # TODO: Supply answer here
    answer = """
    def is_palindrome(word):
    # Remove any whitespace and convert to lowercase
    word = word.replace(" ", "").lower()
    
    # Compare the word with its reversed version
    if word == word[::-1]:
        return True
    else:
        return False

    """
    return answer


def eval_coding_answer(conversation_buf, answer):
    check_answer_template = f"The candidate provided this answer {answer}. Can you rate the answer based on code quality, correctness and code formatting"
    get_answer_scores = f'Extract the scores from above into a csv string?'

    response = conversation_buf(check_answer_template)['response']
    print(response)
    scores = conversation_buf(get_answer_scores)['response']
    print(scores)
    summarize_review = f'Provide a summary of the {response}'
    summary = conversation_buf(summarize_review)['response']
    print(summary)


def run_coding_questions():
    question, test_cases = generate_coding_question(
        conversation_buf, ['Palindrome'], 'hard')
    # question, test_cases = generate_coding_question(
    #    conversation_buf, ['LinkedList', 'HashMap'], 'hard')
    answer = get_coding_answer(conversation_buf, question)
    eval_coding_answer(conversation_buf, answer)


run_coding_questions()
