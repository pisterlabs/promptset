import openai


class PromptGenerator:
    
    def __init__(self, text_data, difficulty):
        self.text = text_data
        self.difficulty_sentences = ''

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
    
    def make_short_answer_prompt(self, number_of_questions, questions_format):
        prompt =  f'''You are helpful Quiz generator
            Please generate a quiz containing {number_of_questions} short answer questions. {self.difficulty_sentences}. The quiz should follow the strict rules below:
    - the quiz must be enclosed in square brackets [ ].
    - The quiz should contain {number_of_questions} questions!!.
    - The quiz should be generated in the following format but the number of questions you generate is {number_of_questions} here is the demo format : """" {questions_format}"""". 
    - The quiz should be enclosed in square brackets [ ], and each question should span only one line!!!! important!!!.
    - The quiz should resemble an academic quiz that helps students prepare for an exam.

    the following keywords are extracted from the book from which the quiz is to be generated. therefore generate the quiz that is related 
    to these topics and their relationships
    """"{self.text}""""
    Thank You for your assistance and for understanding my context!
    '''
        return prompt
    
        

class OpenAi:

    def __init__(self, API_KEY):
        self.api_key = API_KEY
        openai.api_key = self.api_key
        
    def chat(history, query):
        conversation_history = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        for chat in history:
            if chat.is_recieved:
                hist = {'role':'assistant', 'content':chat.text[-500:]}
                conversation_history.append(hist)
            else:
                hist = {'role':'user', 'content':chat.text[-500:]}
                conversation_history.append(hist)
        conversation_history.append({'role':'user','content':query})
        
        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages = conversation_history,
        )
        generated_response = completion.choices[0].message["content"]
        return generated_response
    
    def generate_question(self, prompt):
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
    
    def gudge_short_answer_submission(self, submission):
        feedback_format = """
        [
            what is 1+1?
            1+1 is 2
            you are correct 1+1 is 2. this is a simple mathematical concept ...
            $$$$
            what is the big bang?
            it is a big thing
            your answer for this question is wrong or incomplete. in theoretical physics the big bang is an event that happend billons of years ago ans it is believed to be the cause for the birth od the universe
            $$$$
            what is motion?
            moion is the movement of particles
            your answer for this question seems correct but it is not accurate. you need to improve your explanation. motion is the change of spac coordinate of an object through the passage of time
            ]
            """
        prompt = f"""
        You are helpful quiz judge.
        JUDGE THE FOLLOWING QUIZ OBJECTIVELY 
        the following question answer pairs was submitted by me '''{submission}'''
        when the key of the above object is the question the value is the my answer for that question
        so based on that judge the my answers wheather they are right or wrong and give me corrective feedback and challange my answers for each question.
        your response should be in the following format '''{feedback_format}'''
        NOTICE YOU CAN ONLY RETURN YOUR RESPONSE IN THE GIVEN FORMAT ABOVE: WHERE
        THE FEEDBACK MUST BE ENCLOSED WITH ANGLE BRACKETS [ ] AND FEEDBACK FOR EACH QUESTION IS SEPARATED BY NEW LINE + FOUR DOLLAR SIGHNS I.E $$$$ AND A NEW LINE
        FOR EACH QUESTION THE FEEDBACK ONLY CONTAINS THREE LINES THE FIRST LINE IS THE ORIGINAL QUESTION THE SECOND LINE IS MY ANSWER WITH OUT ANY MODIFICATION AND
        THE THIRD LINE IS YOUR FEEDBACK FOR MY ANSWER IN ADDITION TO FURTHER EXPLANATION
        AND NEVER USE QUOTED WORDS SINCE I WANT TO PARSE THAT TEXT TO JSON. EACH QUESTION MY ANSWER AND YOU FEED BACK SHOUL NOT CONTAIN NEW LINE THEY MUST BE IN A SINGLE LINE
        FOR EACH QUESTION THE FEEDBACK MUST BE THREE LINE AND EACH FEEDBACKS MUST BESEPARATED BY NEW LINE + $$$$ + LEWLINE
        AND YOU MUST OBJECTIVELY JUDGE MY QUIZ. MAKE YOU EXPLANTIONS DETAILED. Do not forget to enclose your response with [ ]
        there must be [ at the start of your response and ] at the end as included in the demo
        
        THANK YOU FOR YOUR ASSISTANCE
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


if __name__ == '__main__':
    from dotenv import load_dotenv
    import os
    load_dotenv()
    from parse_response_v2 import parse_short_answer_submission
    
    BARD_API_KEY = os.getenv("BARD_API_KEY")
    OPEN_AI_API_KEY  = os.getenv('OPEN_AI_API_KEY')  
    OPEN_AI_API_KEY  = os.getenv('OPEN_AI_API_KEY')
    
    open_ai = OpenAi(OPEN_AI_API_KEY)
    my_submissions = {
        "what is time?":"time is the passage of events",
        "what is special relativity": "special relativity is a theory stating that t space time are inseparable",
        "why time passes": "time passes because of energy",
        "what is the meaning of quantum pysics": "i dont know"
    }
    response = open_ai.gudge_short_answer_submission(my_submissions)
    print(response)
    resp = parse_short_answer_submission(response)
    print('-----------------------------------------')
    print(resp)
    
    
    
