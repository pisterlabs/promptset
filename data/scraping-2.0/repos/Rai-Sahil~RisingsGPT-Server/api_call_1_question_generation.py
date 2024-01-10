# api_call_1_question_generation.py

import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
# Set GPT model

# openai.api_key = os.getenv("OPENAI_API_KEY")
# gpt_model = "gpt-3.5-turbo"

openai.api_key = os.environ.get("OPEN_API")
gpt_model = "gpt-4"
# gpt_model = "gpt-3.5-turbo"

def gpt_response_1_question_generation(essay_question, personal_summary_points, essay_length):
    response = openai.ChatCompletion.create(model=gpt_model, temperature=0, messages=[
        ### SYSTEM PROMPT
        {"role": "system",
         "content":
             """
             You are a helpful assistant.
             """
         },

        ### USER PROMPT THAT IS ACTUALLY USED FOR A RESPONSE FROM THE API
        {"role": "user",
         "content":
             f"""

             I need help writing a {essay_length} word essay, I want this to be a heavily personal and truly awe 
             inspiring essay to the resilience of the human spirit. I need help with thinking brainstorming what to 
             write about, can you provide me 10 questions that will help me analyze how to write the essay? These 
             questions have to be very well thoughtout, and I do not want to see any duplicate questions.

             The essay needs to be {essay_length} words long.

             Here is the essay question that I need to answer:
             {essay_question}

             When thinking of questions to ask me, use the essay question to find the proper questions to ask me. I 
             need to answer the essay question, not make a genearl essay. In addition, use these guidelines to help 
             you get that relevant information from me:

             - Keep the focus of your essay narrow and personal, illuminating your character through specific experiences.
             - Ensure that your essay authentically showcases your qualities such as energy, resilience, leadership, passion, inclusivity, and unique outlooks. Write in your own voice.
             - Incorporate elements of creativity, 'showing' not just 'telling' your experiences. Use figurative language where appropriate and maintain a clear structure. Describe sights, smells, tastes, tactile sensations, and sounds as you write. You want the reader to be entirely absorbed in the story you are telling.
             - Think of the typical five paragraph structure for English papers. Your essay should have an introductory paragraph with a thesis/hook, supporting body paragraphs, and a conclusion that ties everything together. Your story might lend itself to six or seven paragraphs instead of five, depending on where the natural narrative breaks lie, and that’s fine. Just make sure it has a clear beginning, middle, and end.
             - Conclude your essay by delivering a key point or insight about yourself, making sure the reader walks away with a strong understanding of who you are."

             Here is a basic introduction to me:
             {personal_summary_points}
             """},
    ])

    # print(response)
    # print(response.choices[0].message.content)

    return response.choices[0].message.content
