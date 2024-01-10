from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import json
import re


def clean_and_convert_to_json(output):
    cleaned_texts = [re.sub(r'^\s*[^:]+:\s*', '', text) for text in output]  
    cleaned_texts = [re.sub(r'\s+', ' ', text.strip()) for text in cleaned_texts]

    json_data = []
    for text in cleaned_texts:
        try:
            json_data.append(text)
        except json.JSONDecodeError:
            pass  # Ignore non-formatted or invalid JSON
    json_array = json.dumps(json_data, indent=4)
    data = json.loads(json_array)
    print(data)
    return data


def MCQGen(num_questions, difficulty_level, topic):

    template = """ 
    You are an AI bot that aids exam setters in generating MCQ questions.
    Below is the information you need to generate.
    You have to generate {num_questions} questions with 4 options as an array, indicating the index position of the correct answer. The questions should be of difficulty level {difficulty_level} and the topic is {topic}.

    Please return the answers in the JSON array format as follows:
    {{
        
        "question" : // a string
        "options" : // an array of 4 strings
        "answer" : // an index position of the correct answer
    }}
    
    Don't forget to add a comma after each question except the last one and make it a valid JSON array.
    
    """
    prompt_template = PromptTemplate(input_variables=["num_questions", "difficulty_level", "topic"],template=template)
    
    chatgpt_chain = LLMChain(llm=OpenAI(temperature=0.75, openai_api_key='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'),prompt=prompt_template)
    output_response = []
    while num_questions > 0:
        questions_to_generate = min(4, num_questions)
        output = chatgpt_chain.predict(num_questions=questions_to_generate, difficulty_level=difficulty_level, topic=topic)

        output_response.append(output)
        num_questions -= questions_to_generate
    
    output = clean_and_convert_to_json(output_response)

    return output
    
if __name__ == "__main__":
    output = MCQGen(4, "easy", "python developer")
    print(output)
  
    

