# Import all dependent libraries

from warnings import filterwarnings
filterwarnings('ignore')

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

from langchain import OpenAI, ConversationChain
from langchain.chat_models import ChatOpenAI

import PyPDF2
import random
import joblib




# Function definitions


def get_context(path, get_page_num = 4):
    """This function will get you the context from the pdf file that you need to generate MCA questions from.

    Args:
        path (str): absolute or relative path to the PDF file 
        get_page_num (int, optional): Extract context from a specified page. Defaults to 4.

    Raises:
        KeyError: If page number out of bounds
        Exception: Any other error

    Returns:
        str: Extracted content from a specified page will be returned as str
    """
    pdf_content = {}

    with open(path, "rb") as file:
        
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)

        for page_num in range(num_pages):
            page = reader.pages[page_num]
            page_content = page.extract_text()
            pdf_content[page_num + 1] = page_content
     
    context = pdf_content.get(get_page_num)
    
    if get_page_num > num_pages:
        raise KeyError(f"Page number doesn't exist in the PDF. Number of Pages in the PDF is {num_pages}")
    elif context:
        return context
    else:
        raise Exception (f"An Error Occured. Check whether your parameters are right")


           
def preprocess_text(text):
    """This funtion will preprocess the context you pass.

    Args:
        text (str): Pass the context from which you want to frame questions

    Returns:
        str: Preprocessed context for framing questions.
    """
    
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        cleaned_sentences.append(' '.join(words))

    return ' '.join(cleaned_sentences)



def generate_mcqs(context, num_questions = 3, total_options = 4, correct_options = 2):
    """Preprocessed text will be tokenized and labelled with respective parts of speech tags. Then the function will use this and pass it to LLM to frame questions based on given template and their corresponding options.

    Args:
        context (str): Preprocessed context from which you want to frame questions.
        num_questions (int, optional): Total number of questions you want the llm to generate. Defaults to 3.
        total_options (int, optional): Total number of options per question. Defaults to 4.
        correct_options (int, optional): Number of correct options you want out of total number of options. Defaults to 2.

    Raises:
        TypeError: If str is not passed

    Returns:
        List of strings: List of strings where each string is a set of question and corresponding options.
    """
    if type(context) != str:
        raise TypeError("Context must be of string datatype")
    else:
        cleaned_text = preprocess_text(context)

        tokenized_text = word_tokenize(cleaned_text)
        tagged_text = pos_tag(tokenized_text)

        question_templates = [
            "What is the main idea behind {}?",
            "Which of the following is wrong about {}?",
            "What can be inferred from the {}?",
            "Which of the following statements is true about {}?",
            "What is the significance of {} in the context?",
        ]

        questions = []

        llm = OpenAI(temperature = 0, max_retries = 1)

        for i in range(num_questions):
            question_template = random.choice(question_templates)

            question_entity = llm.generate(prompts = [f"Give me an entity from the context provided after ':' which fits properly with this {question_template} and choose it in such a way that the answers for the question can be framed within the given context: {context}"])
            question = question_template.format(question_entity)

            right_options = []
            wrong_options = []

            for j in range(total_options - 1):

                right_option = llm.generate(prompts = [f"Give me {correct_options} correct option(s) for this question: {question} as python list of string(s)"])
                wrong_option = llm.generate(prompts = [f"Give me {total_options-correct_options} wrong option(s) for this question: {question} as python list of string(s)"])
                
                right_options.append(right_option)
                wrong_options.append(wrong_option)

            all_options = right_options + wrong_options
            random.shuffle(all_options)

            questions.append((question, all_options))

        return questions

def get_mca_questions(context, num_questions = 3, total_options = 4, correct_options = 2):
    """The function will pass the context provided without any preprocessing to an llm and uses conversation buffer memory which we got by finetuning the model to our needs and frame questions based on given template and their corresponding options using the same conversation chain on every fresh function call.

    Args:
        context (str): Context from which you want to frame questions and corresponding options.
        num_questions (int, optional): Total number of questions you want the llm to generate. Defaults to 3.
        total_options (int, optional): Total number of options per question. Defaults to 4.
        correct_options (int, optional): Number of correct options you want out of total number of options. Defaults to 2.

    Raises:
        TypeError: If str is not passed

    Returns:
        List of strings: List of strings where each string is a set of question and corresponding options.
    """
    if type(context) != str:
        raise TypeError("Context must be of string datatype")
        
    else:
        prompt = f"""
                   context: {context}
                   num_questions: {num_questions}
                   total_options: {total_options}
                   correct_options: {correct_options}
                """
        
        llm = ChatOpenAI(temperature = 0)
        conversation = ConversationChain(llm = llm, memory = joblib.load('convo.pkl'))
        mca_questions = conversation.predict(input = prompt)
        
    return mca_questions