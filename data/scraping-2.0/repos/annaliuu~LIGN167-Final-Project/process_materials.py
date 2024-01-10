import os
import openai


# Part 1: Data Processing


def read_markdown_file(file_path):
   try:
       with open(file_path, 'r', encoding='utf-8') as file:
           content = file.read()
       return content
   except FileNotFoundError:
       print(f"File not found: {file_path}")
       return None


def process_lecture_materials(directory='lectures'):
   topics = {
       "What is Language": "2_what_is_language.mp4.wav.txt",
       "Phonetics": "4_phonetics_1.mp4.wav.txt",
       "Phonology": "6_phonology_1.mp4.wav.txt",
       "Morphology": "8_morphology.mp4.wav.txt",
       "Syntax": "10_syntax1.mov.wav.txt",
       "Semantics/Pragmatics": "14_semantics_and_pragmatics.mp4.wav.txt",
       "Language Families": "16_language_families.mp4.wav.txt"
   }


   processed_materials = {}
   for topic, filename in topics.items():
       file_path = os.path.join(directory, filename)
       lecture_content = read_markdown_file(file_path)
       if lecture_content is not None:
           processed_materials[topic] = lecture_content


   return processed_materials




# Part 2: Question Generation
class QuestionBank:
   def __init__(self, api_key, lecture_materials):
       self.api_key = api_key
       self.lecture_materials = lecture_materials


   # Code to summarize the entire lesson
   # Not used since it heavily uses the API
   def summarize_chunk(self, chunk):
       try:
           response = openai.ChatCompletion.create(
               model="gpt-3.5-turbo",
               messages=[
                   {"role": "system", "content": "This is a text summarization task. Summarize the following text."},
                   {"role": "user", "content": chunk}
               ],
               max_tokens=150,  # Adjust based on your needs for summary length
               api_key=api_key
           )
           return response.choices[0].message.content.strip()
       except Exception as e:
           print(f"An error occurred: {e}")
           return None


   def generate_question(self, topic):
       max_length = 3500
       long_text = lecture_materials[topic]


       # Alternative usage using the API summary; not being used
       # chunks = [long_text[i:i + max_length]
       #           for i in range(0, len(long_text), max_length)]
       # sum_chunks = [str(self.summarize_chunk(chunk)) for chunk in chunks]
       # summary = ''.join(sum_chunks)
       summary = long_text[:max_length]


       try:
           response = openai.ChatCompletion.create(
               model="gpt-3.5-turbo",  # Specify the engine you want to use
               messages=[
                   {"role": "system", "content": "This is a question generation session. Create a good multiple choice question for an exam based on the provided summary. Do not reference the lecture at all."},
                   {"role": "user", "content": f"Create a multiple choice question with four answers about the topic:{topic} using this content: {summary}.\n This questions should be for an exam, and should not reference this specific lecture but rather the topic. Follow the format of this example:\nQ: How many letters in the word 'crazy'?\nA. 8\nB. 16\nC. 5\nD. 18"}
               ],
               max_tokens=150,  # The maximum number of tokens to generate in the response
               api_key=api_key
           )
           question = response.choices[0].message.content.strip()
           return question, summary
       except Exception as e:
           print(f"An error occurred: {e}")
           return "Error generating question"


   def generate_answer(self, question, summary):
       answer = openai.ChatCompletion.create(
           model="gpt-3.5-turbo",  # Specify the engine you want to use
           messages=[{"role": "system", "content": "This is a question and answer session. Answer the question based on the provided summary."},
                     {"role": "user", "content": f"What is the correct answer to {question} based on {summary}? Respond with exactly one letter, and the explanation."}],
           max_tokens=150,  # The maximum number of tokens to generate in the response
           api_key=api_key
       )
       return answer.choices[0].message.content.strip()




# Example usage
# Replace with your actual API key
api_key = "insert_api_key"
lecture_materials = process_lecture_materials()
question_bank = QuestionBank(api_key, lecture_materials)


# Generate a question based on performance
# next_question = question_bank.generate_question(
#     'Introducing Language and Dialect')
# print(next_question[0])


# ans = question_bank.generate_answer(next_question[0], next_question[1])
# print(ans)
