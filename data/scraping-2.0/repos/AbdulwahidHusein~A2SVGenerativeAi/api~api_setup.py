import openai
from bardapi import Bard

class PromptGenerator:
    
    def __init__(self, text_data, difficulty):
        self.text = text_data
        self.difficulty_sentences = ''
        self.prompt = ""
        self.difficulty = difficulty
        if difficulty == 'easy':
            self.difficulty_sentences = 'The questions should be designed with simplicity in mind but logical, making them accessible to individuals who possess basic knowledge of the topic.'
        elif difficulty == 'medium':
            self.difficulty_sentences = 'The questions should offer a moderate level of challenge, suitable for individuals with an average understanding of the topic.'
        else:
            self.difficulty_sentences = 'The questions should be intentionally difficult, requiring a deep comprehension of the topic for anyone attempting to answer them. onlyindividuls having a deeper understanding of the content should answer these questions'
    def make_multiple_choice_prompt(self, number_of_questions, question_format):
        prompt = f'''You are helpful Quiz generator for academic prepouse
            Please generate a quiz containing {number_of_questions} multiple-choice questions. {self.difficulty_sentences}. The quiz should follow the strict rules below:
    - the quiz must be enclosed in square brackets [ ].
    - Each question should consist of a question, four options, the correct option, and an explanation Note The quiz should contain {number_of_questions} questions!!.
    - The quiz should be generated in the following format or demo but the number of questions you generate is {number_of_questions} here is the format : """" {question_format}"""". Note the use of A., B., C., and D., and the placement of the correct option followed by the option letter.
    - The quiz should be enclosed in square brackets [ ], and each question should be separated by four dollar signs $$$$ as indicated in the format. Ensure that the text, options, and explanations for each question are on a single line, even if they are long.
    - The quiz should resemble an academic quiz that helps students prepare for an exam.

    make your quiz very smart and logical that asesses students understandinf of the topic. it should not be a random set of questions without a context.
    the questions must be helpful for studens to understand their topic well the explanations also.
    the following keywords are extracted from the book from which the quiz is to be generated. therefore generate the quiz that is related 
    to these topics and their note try please dont use these keywords separately they are part of some texx book so use them appropirately. 
    """"{self.text}""""
    Thank You for your assistance and for understanding my context!
    '''
        return prompt
    
    def make_short_answer_propmt(self, number_of_questions, question_format):
        prompt  = f'''Dear AI Model,
            Please generate a quiz that consists of {number_of_questions} different short answer questions.{self.difficulty_sentences} Each question should have a unique context. Set the difficulty level to {self.difficulty_sentences}. The questions must be returned in the following format: {question_format}. 

            Please note that each question should not be more than two lines NEVER INSERT NEW LINES WITHIN A SINGLE ITEM THAT ITEM MAY BE A QUESTION OR ANSWER. It is important to ensure that all items are quoted to avoid JSON errors. Do not use quoted words that may interfere with the string's quotation. Remember to quote each item properly.

            The question itself should be a valid JSON format with appropriate quotes and commas. 
            make your quiz very smart and logical that asesses students understandinf of the topic. it should not be a random set of questions without a context.
            the questions must be helpful for studens to understand their topic well the explanations also.
            the following keywords are extracted from the book from which the quiz is to be generated. therefore generate the quiz that is related 
            to these topics and their note try please dont use these keywords separately they are part of some texx book so use them appropirately. 
            """"{self.text}""""
            never forget to return your response in the required format
            Thank You for your assistance and for understanding my context!
            '''
        self.prompt = prompt
        return prompt
        

class OpenAi:

    def __init__(self, API_KEY):
        
        self.api_key = API_KEY
        openai.api_key = self.api_key
        
    def chat(self, history, query, user):
        conversation_history = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        conversation_history.append({'role':'system','content':"assume you are an assistant in quiz website called Quizme in which you generates an interactive quiz from their study material and users ask more calarification about these questions. now you are assisting a person named "+user.first_name + "be freindly with them answer appropirate answers for each of their questions you should only respond to educational prompts"})
        for chat in history:
            if chat.is_received:
                hist = {'role':'assistant', 'content':chat.text}
                conversation_history.append(hist)
            else:
                hist = {'role':'user', 'content':chat.text}
                conversation_history.append(hist)
        
        conversation_history.append({'role':'user','content':user.first_name + ": "+query})
        
        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages = conversation_history,
        )
        generated_response = completion.choices[0].message["content"]
        return generated_response
    
    def generate_question(self, prompt):
        prompt_token_length = len([w for w in prompt.split()])
        max_tokens = 3500 - prompt_token_length
        parameters = {
            'model': 'text-davinci-003',  # Choose the model you want to use
            'prompt': prompt,
            'max_tokens': max_tokens,  # Adjust the length of the generated reply as needed
            'temperature': 0.5,  # Adjust the temperature to control the randomness of the output
            'n': 1,  # Generate a single reply
            'stop': None,  
            }
        response = openai.Completion.create(**parameters)
        reply = response.choices[0].text.strip()
        return reply
    def gudge_short_answer_submission(self, submission):
        feedback_format = """
        [
            what is 1+1?
            1+1 is 2
            you are correct 1+1 is 2. this is a simple mathematical concept ...
            $$$$
            what is the big bang?
            it is a big thing
            your answer for this question is wrong or incomplete. in theoretical physics the big bang is an event that happend billons of years ago and it is believed to be the cause for the birth od the universe while answering these kinds of questions you should give detailed explanations to show your understanding od the question
            $$$$
            what is motion?
            moion is the movement of particles
            your answer for this question seems correct but it is not accurate. you need to improve your explanation. motion is the change of spac coordinate of an object through the passage of time.
            ]
            """
        prompt = f"""
        Dear AI Quiz Judge,

        I kindly request your objective evaluation of the following quiz submissions. The question-answer pairs provided in the submission are as follows: '''{submission}'''. I would appreciate your assessment of my answers for each question to determine their accuracy. Please provide corrective feedback and challenge my answers accordingly.

        To facilitate the evaluation process, please structure your response using the specified format: '''{feedback_format}'''. Please note that the feedback format serves as a demonstration template, and I kindly ask you to provide your feedback for my earlier quiz submission.

        Please be aware that the feedback format requires specific guidelines to ensure proper parsing. The response should be enclosed within angle brackets [ ], and feedback for each question should be separated by a new line followed by four dollar signs ($$$$) and another new line. Each feedback for a question should consist of three lines: the original question, my unmodified answer, and your feedback with additional explanation. Please refrain from using quotation marks in the feedback, as the text will be parsed into JSON. Each question, my answer, and your feedback should be in a single line without any new lines.

        It is crucial that you provide an objective evaluation of my quiz submission. Please offer detailed explanations in your feedback. Remember to enclose the entire response within [ ] brackets, with an opening [ at the beginning and a closing ] at the end, as demonstrated in the template.

        Thank you for your valuable assistance.
        Thank you for your response and understanding my context!!!
        """
        prompt_token_length = len([w for w in prompt.split()])
        max_tokens = 3100 - prompt_token_length 
        parameters = {
            'model': 'text-davinci-003',  
            
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': 0.7,  # Adjust the temperature to control the randomness of the output
            'n': 1,  # Generate a single reply
            'stop': None,  # You can specify a stop sequence to control the length of the completion
        }
        response = openai.Completion.create(**parameters)
        reply = response.choices[0].text.strip()
        return reply
        

 
class BardEx:
    def __init__(self, API_KEY) -> None:
        self.api_key = API_KEY
        self.bard = Bard(token=self.api_key)
        
    def get_answer(self, question):
        answer = self.bard.get_answer(question)
        return answer['content']
