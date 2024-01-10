
import openai

# Set the API key
openai.api_key = ""

def evaluate_essay(essay_content):
    """Evaluate the IELTS essay using OpenAI's GPT model."""
    prompt_for_evaluation = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that evaluates IELTS essays. \
                                            Please evaluate the following IELTS essay and provide an overall score as well as scores based on the four criteria:\
                                            Task Achievement, Coherence and Cohesion, Lexical Resource, and Grammatical Range and Accuracy. \
                                            Provide an overall comment, scores, reasoning, and suggestions for improvement for each criterion. \
                                            Return all the results in Chinese \n\n \
                                            The student's IELTS writting is as follows:"},
            {"role": "user", "content": essay_content}
        ]
    }
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            **prompt_for_evaluation,
            temperature=0.0,)
    except Exception as e:
        return {"error": str(e)}
    
    # Extract the content of the assistant's response
    assistant_message = response['choices'][0]['message']['content']
    
    return assistant_message

# def evaluate_essay(essay_content, model_name="gpt-4", max_tokens=7000, temperature=0.1, top_p=1, frequency_penalty=0.0, presence_penalty=0.6):
#     """Evaluate the IELTS essay using OpenAI's GPT model."""
    
#     # Define the prompt template
#     prompt_template = (
#         "Please evaluate the following IELTS essay and provide an overall score as well as scores "
#         "based on the four criteria: Task Achievement, Coherence and Cohesion, Lexical Resource, "
#         "and Grammatical Range and Accuracy. Provide an overall comment, scores, reasoning, "
#         "and suggestions for improvement for each criterion. \n\nReturn all the results in "
#         "Chinese and in JSON format. \n\nThe essay is as follows:\n\n{}"
#     )
    
#     prompt = prompt_template.format(essay_content)

#     try:
#         # Generate a response
#         response = openai.ChatCompletion.create(
#             model=model_name,
#             prompt=prompt,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             frequency_penalty=frequency_penalty,
#             presence_penalty=presence_penalty,
#         )
        
#         # Extract the assistant's message and return it
#         assistant_message = response['choices'][0]['text']
#         return assistant_message.strip()
    
#     except Exception as e:
#         return {"error": str(e)}

